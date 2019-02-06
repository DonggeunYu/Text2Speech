import argparse
import os
from multiprocessing import cpu_count
from hparams import hparams
from datasets import preprocessor
from tqdm import tqdm

def preprocess(args, input_folders, out_dir, hparams):
    mel_dir = os.path.join(out_dir, 'mels')
    wav_dir = os.path.join(out_dir, 'audio')
    linear_dir = os.path.join(out_dir, 'linear')
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(linear_dir, exist_ok=True)

    metadata = preprocessor.build_from_path(hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs,
                                            tqdm=tqdm)
    write_metadata(metadata, out_dir)

def run_preprocess(args, hparams):
    input_folders = [args.dataset]
    output_folder = args.output

    preprocess(args, input_folders, output_folder, hparams)

def main():
    print('initializing preprocessing...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', default='')
    parser.add_argument('--dataset', default='kss')
    parser.add_argument('--output', default='training_data')
    parser.add_argument('--n_jobs', type=int, default=cpu_count())

    args = parser.parse_args()
    modified_hp = hparams.parse(args.hparams)

    run_preprocess(args, modified_hp)

if __name__ == '__main__':
    main()