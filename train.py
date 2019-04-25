import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from hparams import hparams
from tacotron.tacotron import Tacotron

from utils import prepare_dirs, ValueWindow, str2bool
from utils import infolog

from utils.audio import save_wav, inv_spectrogram

from datasets.datafeeder_tacotron import DataFeederTacotron
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

log = infolog.log

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

def train(log_dir, config):
    config.data_path = config.data_paths

    data_dirs = config.data_paths
    num_speakers = len(data_dirs)
    config.num_test = config.num_test_per_speaker * num_speakers  # 2*1

    if num_speakers > 1 and hparams.model_type not in ["deepvoice", "simple"]:
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

    train_loader = DataLoader(dataset=train_feeder, batch_size=32, shuffle=False, num_workers=os.cpu_count())

    model = Tacotron(hparams, train_loader)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', default='./data/kss')
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--log_dir', default='logdir-tacotron')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_test_per_speaker', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=100)
    parser.add_argument('--skip_path_filter', type=str2bool, default=False, help='Use only for debugging')

    parser.add_argument('--initialize_path', default=None)

    config = parser.parse_args()
    config.data_paths = config.data_paths.split(',')
    hparams.update({"num_speakers": len(config.data_paths)})

    prepare_dirs(config, hparams)

    log_path = os.path.join(config.model_dir, 'train.log')

    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    print(config.data_paths)

    if config.load_path is not None and config.initialize_path is not None:
        raise Exception(" [!] Only one of load_path and initialize_path should be set")

    train(config.model_dir, config)
if __name__ == '__main__':
    main()