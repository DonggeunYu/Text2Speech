import argparse
import os
import importlib
from tqdm import tqdm
from hparams import hparams, hparams_debug_string
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def preprocess(mod, out_dir, in_dir, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata = mod.build_from_path(hparams, in_dir, out_dir, num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for lines in metadata:
            f.write('|'.join([str(i) for i in lines]) + '\n')
    mel_frames = sum([int(lines[4]) for lines in metadata])
    timesteps = sum([int(lines[3]) for lines in metadata])
    sr = hparams.sample_rate
    hours = timesteps / sr / 3600
    print(metadata)
    print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(len(metadata), mel_frames, timesteps, hours))
    print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
    print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
    print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))

if __name__ == "__main__":
    """
    Args:
        --name: name of dataset
        --num_workers: how many use the worker?
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='kss')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())

    args = parser.parse_args()

    name = args.name
    out_dir = './data/' + name
    in_dir = './datasets/' + name
    num_workers = args.num_workers

    if name == None or out_dir == None:
        print("Error: argument is None")
        raise ValueError

    print(hparams_debug_string())

    print('-' * 50)
    print("Sampling frequency: {}".format(hparams.sample_rate))
    print("Num worker: {}".format(num_workers))
    print('-' * 50)
    print('')


    mod = importlib.import_module('datasets.{}'.format(name))
    preprocess(mod, out_dir, in_dir, num_workers)