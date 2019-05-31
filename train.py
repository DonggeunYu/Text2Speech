import argparse
import os
import time
import numpy as np
import torch
import math
from torch import optim
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.distributed as dist
from hparams import hparams
from tacotron.tacotron import Tacotron
from text.symbols import symbols
from os.path import join, dirname
from tqdm import tqdm
from utils.logger import Tacotron2Logger


from utils import prepare_dirs, ValueWindow, str2bool
from utils import infolog

from utils.audio import save_wav, inv_spectrogram

from datasets.datafeeder_tacotron import DataFeederTacotron
import warnings
import sys
warnings.simplefilter(action='ignore', category=FutureWarning)

log = infolog.log

fs = hparams['sample_rate']
global_step = 0
global_epoch = 0

def save_and_plot_fn(args, log_dir, step, loss, prefix):
    idx, (seq, spec, align) = args

    audio_path = os.path.join(log_dir, '{}-step-{:09d}-audio{:03d}.wav'.format(prefix, step, idx))
    align_path = os.path.join(log_dir, '{}-step-{:09d}-align{:03d}.png'.format(prefix, step, idx))

    waveform = inv_spectrogram(spec.T,hparams)
    save_wav(waveform, audio_path,hparams.sample_rate)

    info_text = 'step={:d}, loss={:.5f}'.format(step, loss)
    if 'korean_cleaners' in [x.strip() for x in hparams.cleaners.split(',')]:
        log('Training korean : Use jamo')
        plot.plot_alignment( align, align_path, info=info_text, text=sequence_to_text(seq,skip_eos_and_pad=True, combine_jamo=True), isKorean=True)
    else:
        log('Training non-korean : X use jamo')
        plot.plot_alignment(align, align_path, info=info_text,text=sequence_to_text(seq,skip_eos_and_pad=True, combine_jamo=False), isKorean=False)

def save_and_plot(sequences, spectrograms,alignments, log_dir, step, loss, prefix):

    fn = partial(save_and_plot_fn,log_dir=log_dir, step=step, loss=loss, prefix=prefix)
    items = list(enumerate(zip(sequences, spectrograms, alignments)))

    parallel_run(fn, items, parallel=False)
    log('Test finished for step {}.'.format(step))

def _pad(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x

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


def _pad_input(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=0)


def _pad_target(t, length):
    # t: 2 dim array. ( xx, num_mels) ==> (length,num_mels)
    return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=0)  # (169, 80) ==> (length, 80)

###
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

def save_states(global_step, mel_outputs, linear_outputs, attn, y,
                input_lengths, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))

    # idx = np.random.randint(0, len(input_lengths))
    idx = min(1, len(input_lengths) - 1)
    input_length = input_lengths[idx]

    # Alignment
    path = join(checkpoint_dir, "step{}_alignment.png".format(
        global_step))
    # alignment = attn[idx].cpu().data.numpy()[:, :input_length]
    alignment = attn[idx].cpu().data.numpy()
    save_alignment(path, alignment)

    # Predicted spectrogram
    path = join(checkpoint_dir, "step{}_predicted_spectrogram.png".format(
        global_step))
    linear_output = linear_outputs[idx].cpu().data.numpy()
    save_spectrogram(path, linear_output)

    # Predicted audio signal
    signal = audio.inv_spectrogram(linear_output.T)
    path = join(checkpoint_dir, "step{}_predicted.wav".format(
        global_step))
    audio.save_wav(signal, path)

    # Target spectrogram
    path = join(checkpoint_dir, "step{}_target_spectrogram.png".format(
        global_step))
    linear_output = y[idx].cpu().data.numpy()
    save_spectrogram(path, linear_output)


def collate_fn(batch):
    """Create batch"""
    reduction_factor = 5
    # batch data: (input_data, loss_coeff, mel_target, linear_target, self.data_dir_to_id[data_dir], len(linear_target))
    inputs = _prepare_inputs([x[0] for x in batch])  # batch에 있는 data들 중, 가장 긴 data의 길이에 맞게 padding한다.
    inputs = torch.LongTensor(inputs)

    input_lengths = np.asarray([len(x[0]) for x in batch])  # batch_size, [37, 37, 32, 32, 38,..., 39, 36, 30]
    input_lengths = torch.from_numpy(input_lengths)

    loss_coeff = np.asarray([x[1] for x in batch], dtype=np.float32)  # batch_size, [1,1,1,,..., 1,1,1]
    loss_coeff = torch.LongTensor(loss_coeff)

    mel_targets = _prepare_targets([x[2] for x in batch],
                                   reduction_factor)  # ---> (32, 175, 80) max length는 reduction_factor의  배수가 되도록
    mel_targets = torch.FloatTensor(mel_targets)

    linear_targets = _prepare_targets([x[3] for x in batch],
                                      reduction_factor)  # ---> (32, 175, 1025)  max length는 reduction_factor의  배수가 되도록
    linear_targets = torch.FloatTensor(linear_targets)

    stop_token_targets = _prepare_stop_token_targets([x[4] for x in batch], reduction_factor)
    stop_token_targets = torch.LongTensor(stop_token_targets)

    if len(batch[0]) == 7:  # is_multi_speaker = True인 경우
        speaker_id = np.asarray([x[5] for x in batch], dtype=np.int32)  # speaker_id로 list 만들기
        speaker_id = torch.LongTensor(speaker_id)
        return (inputs, input_lengths, loss_coeff, mel_targets, linear_targets, stop_token_targets, speaker_id)
    else:
        return (inputs, input_lengths, loss_coeff, mel_targets,
                linear_targets, stop_token_targets)

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

    train_feeder = DataFeederTacotron(data_dirs, hparams, config, 32, data_type='train',
                                      batch_size=config.batch_size)
    test_feeder = DataFeederTacotron(data_dirs, hparams, config, 8, data_type='test', batch_size=config.num_test)

    train_loader = DataLoader(dataset=train_feeder, batch_size=32, shuffle=False,
                              collate_fn=collate_fn, num_workers=1, pin_memory=True)

    num_speakers = len(config.data_paths)
    model = Tacotron(hparams, len(symbols), num_speakers=num_speakers)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.Adam(model.parameters(), lr=hparams['initial_learning_rate'],
                           betas=(hparams['adam_beta1'], hparams['adam_beta2']))

    # Train!
    try:
        train(model, train_loader, test_feeder, optimizer,
              init_lr=hparams['initial_learning_rate'],
              checkpoint_dir=config.log_dir,
              checkpoint_interval=config.checkpoint_interval,
              nepochs=hparams['num_steps'],
              clip_thresh=1.0,
              config=config, multi_speaker=multi_speaker)
    except KeyboardInterrupt:
        save_checkpoint(
            model, optimizer, global_step, config.log_dir, global_epoch)

    print("Finished")
    sys.exit(0)

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_priority_freq = int(3000 / (hparams['sample_rate'] * 0.5) * hparams['num_freq'])

    with torch.no_grad():
        val_loader = DataLoader(valset, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, (inputs, input_lengths, loss_coeff, mel_targets, linear_targets, stop_token_target, speaker_id) in enumerate(val_loader):
            inputs, loss_coeff, mel_targets, linear_targets, stop_token_target, speaker_id = \
                inputs.to(device), loss_coeff.to(device), mel_targets.to(device), \
                linear_targets.to(device), stop_token_target.to(device), speaker_id.to(device)

            sorted_lengths, indices = torch.sort(
                input_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().numpy()
            sorted_lengths = torch.LongTensor(sorted_lengths)
            sorted_lengths = sorted_lengths.to(device)
            sorted_lengths = Variable(sorted_lengths)

            inputs, loss_coeff, mel_targets, linear_targets, stop_token_target, speaker_id = \
                inputs[indices], loss_coeff[indices], mel_targets[indices], \
                linear_targets[indices], stop_token_target[indices], speaker_id[indices]
            inputs, loss_coeff, mel_targets, linear_targets, stop_token_target, speaker_id = \
                Variable(inputs), Variable(loss_coeff), Variable(mel_targets), \
                Variable(linear_targets), Variable(stop_token_target), Variable(speaker_id)
            y_pred = model(multi_speaker, inputs, sorted_lengths, loss_coeff, mel_targets, linear_targets,
                           stop_token_target, speaker_id)

            mel_loss = criterion(y_pred[0], mel_targets)
            linear_loss = torch.abs(y_pred[1] - linear_targets)
            linear_loss = 0.5 * torch.mean(linear_loss) + 0.5 * torch.mean(linear_loss[:, :n_priority_freq, :])
            loss = mel_loss + linear_loss
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    print("Validation loss {}: {:9f}  ".format(iteration, reduced_val_loss))
    logger.log_validation(reduced_val_loss, model, (mel_targets, linear_targets), y_pred, iteration)

def train(model, data_loader, test_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0, config=None, multi_speaker=None):
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
    #if torch.cuda.device_count() > 1:
        #print("Use", torch.cuda.device_count(), "GPUs")
        #model = nn.DataParallel(model)

    criterion = nn.L1Loss()

    n_priority_freq = int(3000 / (hparams['sample_rate'] * 0.5) * hparams['num_freq'])

    multi_speaker = torch.LongTensor(multi_speaker)
    multi_speaker = Variable(multi_speaker.to(device))

    if multi_speaker.size(0) > 1: # Multi-Speaker
        for epoch in range(epoch_offset, hparams['num_steps']):
            print("Epoch: {}".format(epoch))
            running_loss = 0.
            for step, (inputs, input_lengths, loss_coeff, mel_targets, linear_targets, stop_token_target, speaker_id)\
                    in tqdm(enumerate(data_loader)):
                start = time.perf_counter()
                current_lr = _learning_rate_decay(init_lr, iteration)

                inputs, loss_coeff, mel_targets, linear_targets, stop_token_target, speaker_id = \
                    inputs.to(device), loss_coeff.to(device), mel_targets.to(device), \
                    linear_targets.to(device), stop_token_target.to(device), speaker_id.to(device)

                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                optimizer.zero_grad()

                # Sort batch data by input_lengths
                sorted_lengths, indices = torch.sort(
                    input_lengths.view(-1), dim=0, descending=True)
                sorted_lengths = sorted_lengths.long().numpy()
                sorted_lengths = torch.LongTensor(sorted_lengths)
                sorted_lengths = sorted_lengths.to(device)
                sorted_lengths = Variable(sorted_lengths)

                inputs, loss_coeff, mel_targets, linear_targets, stop_token_target, speaker_id = \
                    inputs[indices], loss_coeff[indices], mel_targets[indices],\
                    linear_targets[indices], stop_token_target[indices], speaker_id[indices]
                inputs, loss_coeff, mel_targets, linear_targets, stop_token_target, speaker_id = \
                    Variable(inputs), Variable(loss_coeff), Variable(mel_targets), \
                    Variable(linear_targets), Variable(stop_token_target), Variable(speaker_id)
                y_pred = model(multi_speaker, inputs, sorted_lengths, loss_coeff, mel_targets, linear_targets, stop_token_target, speaker_id)

                # Loss
                mel_loss = criterion(y_pred[0], mel_targets)
                linear_loss = torch.abs(y_pred[1] - linear_targets)
                linear_loss = 0.5 * torch.mean(linear_loss) + 0.5 * torch.mean(linear_loss[:, :n_priority_freq, :])
                loss = mel_loss + linear_loss
                #loss = loss.to(device)

                loss.backward()

                #grad_norm = torch.nn.utils.clip_grad_norm_(
                    #model.parameters(), 1.0)

                optimizer.step()

                #duration = time.perf_counter() - start
                print("Loss {}".format(
                    loss))

                iteration += 1
            if (epoch % config.checkpoint_interval == 0):
                checkpoint_path = os.path.join(
                    config.checkpoint_path, "checkpoint_{}".format(epoch))
                save_checkpoint(model, optimizer, learning_rate, epoch,
                                checkpoint_path)

                validate(model, criterion, test_loader, iteration,
                         config.batch_size, collate_fn, logger, multi_speaker)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', default='./data/kss,./data/kss1')
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--checkpoint_file', default='./checkpoint_path/checkpoint_2')
    parser.add_argument('--log_dir', default='logdir-tacotron')
    parser.add_argument('--wav_dir', default='./wav/')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint_path/')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_test_per_speaker', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=100)
    parser.add_argument('--skip_path_filter', type=str2bool, default=False, help='Use only for debugging')

    parser.add_argument('--initialize_path', default=None)

    parser.add_argument('--test_interval', type=int, default=500)  # 500
    parser.add_argument('--checkpoint_interval', type=int, default=2)  # 2000

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

    if config.load_path is not None and config.initialize_path is not None:
        raise Exception(" [!] Only one of load_path and initialize_path should be set")

    if not os.path.exists(config.checkpoint_path):
        os.mkdir(config.checkpoint_path)

    train_init(config.model_dir, config, multi_speaker)

if __name__ == '__main__':
    main()