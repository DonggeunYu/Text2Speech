import argparse
import os
import time
import numpy as np
import torch
from torch import optim
from torch import nn as nn
from torch.utils.data import DataLoader
from hparams import hparams
from tacotron.tacotron import Tacotron
from text.symbols import symbols
from os.path import join, dirname
from tqdm import tqdm

from utils import prepare_dirs, ValueWindow, str2bool
from utils import infolog

from utils.audio import save_wav, inv_spectrogram

from datasets.datafeeder_tacotron import DataFeederTacotron
import tensorboard_logger
from tensorboard_logger import log_value
import warnings
import sys
warnings.simplefilter(action='ignore', category=FutureWarning)

log = infolog.log

global_step = 0
global_epoch = 0
speaker_id = True
use_cuda = torch.cuda.is_available()


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

def add_stats(model, model2=None, scope_name='train'):
    '''
    :param model: first model
    :param model2: second model
    :param scope_name:
    :return: Loss differences between model 1 and model 2
    '''
    with tf.variable_scope(scope_name) as scope:
        summaries = [
                tf.summary.scalar('loss_mel', model.mel_loss),
                tf.summary.scalar('loss_linear', model.linear_loss),
                tf.summary.scalar('loss', model.loss_without_coeff),
        ]

        if scope_name == 'train':
            gradient_norms = [tf.norm(grad) for grad in model.gradients if grad is not None]

            summaries.extend([
                    tf.summary.scalar('learning_rate', model.learning_rate),
                    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms)),
            ])

    if model2 is not None:
        with tf.variable_scope('gap_test-train') as scope:
            summaries.extend([
                    tf.summary.scalar('loss_mel',
                            model.mel_loss - model2.mel_loss),
                    tf.summary.scalar('loss_linear',
                            model.linear_loss - model2.linear_loss),
                    tf.summary.scalar('loss',
                            model.loss_without_coeff - model2.loss_without_coeff),
            ])

    return tf.summary.merge(summaries)

def _pad(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x

def train_init(log_dir, config):
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
                              num_workers=os.cpu_count(), pin_memory=True)
    for step, (input_data, input_data_len, loss_coeff, mel_target, linear_target, stop_token_target, s_id
               , linear_target_len) in enumerate(train_loader):
        pass
    num_speakers = len(config.data_paths)
    model = Tacotron(hparams, len(symbols), num_speakers=num_speakers)

    optimizer = optim.Adam(model.parameters(), lr=hparams['initial_learning_rate'],
                           betas=(hparams['adam_beta1'], hparams['adam_beta2']))

    checkpoint_path = config.checkpoint_path
    # Load checkpoint
    if checkpoint_path:
        print("Load checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            global_step = checkpoint["global_step"]
            global_epoch = checkpoint["global_epoch"]
        except:
            print("Error load checkpoint")
            exit()

    # Setup tensorboard logger
    tensorboard_logger.configure("log/run-test")

    # Train!
    try:
        train(model, train_loader, optimizer,
              init_lr=hparams['initial_learning_rate'],
              checkpoint_dir=config.log_dir,
              checkpoint_interval=config.checkpoint_interval,
              nepochs=hparams['num_steps'],
              clip_thresh=1.0)
    except KeyboardInterrupt:
        save_checkpoint(
            model, optimizer, global_step, config.log_dir, global_epoch)

    print("Finished")
    sys.exit(0)

def _learning_rate_decay(init_lr, global_step):
    warmup_steps = 4000.0
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(
        step * warmup_steps**-1.5, step**-0.5)
    return lr

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{}.pth".format(global_step))
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def train(model, data_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    model.train()
    if use_cuda:
        model = model.cuda()
    linear_dim = model.linear_dim

    criterion = nn.L1Loss()

    global global_step, global_epoch

    if speaker_id: # Multi-Speaker
        while global_epoch < nepochs:
            running_loss = 0.
            for step, (input_data, input_data_len, loss_coeff, mel_target, linear_target, stop_token_target, s_id
                       , linear_target_len) in tqdm(enumerate(data_loader)):
                current_lr = _learning_rate_decay(init_lr, global_step)
                for params_group in optimizer.param_groups:
                    params_group['lr'] = current_lr

                optimizer.zero_grad()

                sorted_lengths, indices = torch.sort(input_data_len)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', default='./data/kss1,./data/kss2')
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--log_dir', default='logdir-t2s')
    parser.add_argument('--checkpoint_path', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_test_per_speaker', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=100)
    parser.add_argument('--skip_path_filter', type=str2bool, default=False, help='Use only for debugging')

    parser.add_argument('--initialize_path', default=None)

    parser.add_argument('--test_interval', type=int, default=500)  # 500
    parser.add_argument('--checkpoint_interval', type=int, default=2000)  # 2000

    config = parser.parse_args()
    config.data_paths = config.data_paths.split(',')
    hparams.update({"num_speakers": len(config.data_paths)})
    global speaker_id
    speaker_id = True if len(config.data_paths) > 1 else False
    prepare_dirs(config, hparams)

    log_path = os.path.join(config.model_dir, 'train.log')

    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    print(config.data_paths)

    if config.load_path is not None and config.initialize_path is not None:
        raise Exception(" [!] Only one of load_path and initialize_path should be set")

    train_init(config.model_dir, config)

if __name__ == '__main__':
    main()