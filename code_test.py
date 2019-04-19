import argparse
import os
import time
import tensorflow as tf

from hparams import hparams
from tacotron import create_model, get_most_recent_checkpoint

from utils import prepare_dirs, ValueWindow
from utils import infolog

from utils.audio import save_wav, inv_spectrogram

from datasets.datafeeder_tacotron import DataFeederTacotron
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

log = infolog.log

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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', default='./data/kss')
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--log_dir', default='logdir-tacotron')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_test_per_speaker', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=100)

    parser.add_argument('--initialize_path', default=None)

    config = parser.parse_args()
    config.data_paths = config.data_paths.split(',')
    hparams.update({"num_speakers": len(config.data_paths)})

    prepare_dirs(config, hparams)

    log_path = os.path.join(config.model_dir, 'train.log')

    tf.random.set_seed(config.random_seed)
    print(config.data_paths)

    if config.load_path is not None and config.initialize_path is not None:
        raise Exception(" [!] Only one of load_path and initialize_path should be set")

    train(config.model_dir, config)
if __name__ == '__main__':
    main()