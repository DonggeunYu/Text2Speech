import os
import sys
import time
import torch
import warnings
import argparse
import numpy as np

from torch import optim
from utils import infolog
from torch import nn as nn
from hparams import hparams
from text.symbols import symbols
from torch.autograd import Variable
from tacotron.tacotron import Tacotron
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils.logger import Tacotron2Logger
from utils import prepare_dirs, str2bool
from tacotron.loss_function import Tacotron2Loss
from utils.data_utils import TextMelCollate, TextMelLoader
from datasets.datafeeder_tacotron import DataFeederTacotron

warnings.simplefilter(action='ignore', category=FutureWarning)
summary = SummaryWriter()

log = infolog.log

fs = hparams['sample_rate']
global_step = 0
global_epoch = 0

def _pad(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)

def _prepare_inputs(inputs):  # inputs: batch 길이 만큼의 list
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad(x, max_len) for x in inputs])

def _prepare_targets(targets, alignment):
    # targets: shape of list [ (162,80) , (172, 80), ...]
    max_len = max((len(t) for t in targets)) + 1
    return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])

def _prepare_stop_token_targets(targets, alignment):
    max_len = max((len(t) for t in targets)) + 1
    return np.stack([_pad_stop_token_target(t, _round_up(max_len, alignment)) for t in targets])

def _pad_target(t, length):
    # t: 2 dim array. ( xx, num_mels) ==> (length,num_mels)
    return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=0)  # (169, 80) ==> (length, 80)

def _pad_stop_token_target(t, length):
    return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=1)

def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder


def _learning_rate_decay(init_lr, global_step):
    warmup_steps = 4000.0
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(
        step * warmup_steps**-1.5, step**-0.5)
    return lr

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def train_init(log_dir, config, multi_speaker):
    config.data_path = config.data_paths

    data_dirs = config.data_paths
    num_speakers = len(data_dirs)
    config.num_test = config.num_test_per_speaker * num_speakers  # 2*1

    if num_speakers > 1 and hparams['model_type'] not in ["deepvoice", "simple"]:
        raise Exception("[!] Unkown model_type for multi-speaker: {}".format(config.model_type))

    checkpoint_path = os.path.join(log_dir, 'model.ckpt')

    log('-'*50)
    log('-'*50)
    log(' [*] Checkpoint path: %s' % checkpoint_path)
    log(' [*] Loading training data from: %s' % data_dirs)
    log(' [*] Using model: %s' % config.model_dir)  # 'logdir-tacotron\\moon_2018-08-28_13-06-42'

    trainset = TextMelLoader(config.data_paths, hparams, 'train')
    valset = TextMelLoader(config.data_paths, hparams, 'val')
    collate_fn = TextMelCollate(hparams['n_frames_per_step'])

    train_loader = DataLoader(trainset, num_workers=0, shuffle=True,
                              batch_size=hparams['batch_size'], pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    num_speakers = len(config.data_paths)
    model = Tacotron(hparams, len(symbols), num_speakers=num_speakers)

    optimizer = optim.Adam(model.parameters(), lr=hparams['initial_learning_rate'],
                           betas=(hparams['adam_beta1'], hparams['adam_beta2']))

    # Train!
    try:
        train(model, train_loader, valset, optimizer,
              init_lr=hparams['initial_learning_rate'],
              checkpoint_dir=config.log_dir,
              checkpoint_interval=config.checkpoint_interval,
              nepochs=hparams['tacotron_decay_steps'],
              clip_thresh=1.0,
              config=config, multi_speaker=multi_speaker,
              collate_fn=collate_fn)
    except KeyboardInterrupt:
        save_checkpoint(
            model, optimizer, global_step, config.log_dir, global_epoch)

    print("Finished")
    sys.exit(0)

def load_checkpoint(checkpoint_path, model, optimizer):
    print(checkpoint_path)
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration

def prepare_directories_and_logger(log_directory):
    if not os.path.isdir(log_directory):
        os.mkdir(log_directory)
    logger = Tacotron2Logger(os.path.join(log_directory))

    return logger

def validate(model, criterion, valset, iteration, batch_size,
             collate_fn, logger, multi_speaker):
    """Handles all the validation scoring and printing"""
    model.eval()

    n_priority_freq = int(3000 / (hparams['sample_rate'] * 0.5) * hparams['num_freq'])

    with torch.no_grad():
        val_loader = DataLoader(valset, num_workers=0,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    print("Validation loss {:9f}  ".format(loss))
    logger.log_validation(reduced_val_loss, model, y, y_pred, iteration)

def train(model, data_loader, test_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0, config=None, multi_speaker=None, collate_fn=None):
    iteration = 0
    epoch_offset = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.checkpoint_file is not None:
        model, optimizer, _learning_rate, epoch_offset = load_checkpoint(
                config.checkpoint_file, model, optimizer)

        epoch_offset += 1  # next iteration is iteration + 1
        #epoch_offset = max(0, int(iteration / len(data_loader)))

    logger = prepare_directories_and_logger(config.log_dir)

    learning_rate = hparams['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams['weight_decay'], betas=(0.9, 0.999))

    model.train()
    model.to(device)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        config.logger_path)

    n_priority_freq = int(3000 / (hparams['sample_rate'] * 0.5) * hparams['num_freq'])

    multi_speaker = torch.LongTensor(multi_speaker)
    multi_speaker = Variable(multi_speaker.to(device))

    if multi_speaker.size(0) > 1: # Multi-Speaker
        for epoch in range(epoch_offset, nepochs):
            print("Epoch: {}".format(epoch))
            running_loss = 0.
            for batch in data_loader:
                start = time.perf_counter()
                current_lr = _learning_rate_decay(init_lr, iteration)


                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                optimizer.zero_grad()
                model.zero_grad()

                x, y = model.parse_batch(batch)
                y_pred = model(x)

                loss = criterion(y_pred, y)
                loss.backward()

                optimizer.step()

                duration = time.perf_counter() - start
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)

                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, loss, grad_norm, duration))

                iteration += 1

                logger.log_training(
                    loss, grad_norm, learning_rate, duration, iteration)

                if (iteration % config.checkpoint_interval == 0):
                    checkpoint_path = os.path.join(
                        config.checkpoint_path, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                checkpoint_path)

                    validate(model, criterion, test_loader, iteration,
                            config.batch_size, collate_fn, logger, multi_speaker)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', default='./datasets/kss/,./datasets/kss/')
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--checkpoint_file', default=None) #'./checkpoint_path/checkpoint_2'
    parser.add_argument('--log_dir', default='logdir-tacotron')
    parser.add_argument('--wav_dir', default='./wav/')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint_path/')
    parser.add_argument('--logger_path', default='./logger/')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_test_per_speaker', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--skip_path_filter', type=str2bool, default=False, help='Use only for debugging')

    parser.add_argument('--checkpoint_interval', type=int, default=1000)  # 2000

    parser.add_argument('--n_gpus', type=int, default=torch.cuda.device_count())

    config = parser.parse_args()
    config.data_paths = config.data_paths.split(',')
    hparams.update({"num_speakers": len(config.data_paths)})
    multi_speaker = len(config.data_paths)
    prepare_dirs(config, hparams)

    log_path = os.path.join(config.model_dir, 'train.log')

    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    print(config.data_paths)

    if not os.path.exists(config.checkpoint_path):
        os.mkdir(config.checkpoint_path)

    train_init(config.model_dir, config, multi_speaker)

if __name__ == '__main__':
    main()