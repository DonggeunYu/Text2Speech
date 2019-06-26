import random
import numpy as np
import torch
import torch.utils.data

import utils.layers as layers
from text import text_to_sequence
from librosa.core import load


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def load_wav_to_torch(full_path, sr):
    data, sampling_rate = load(full_path, sr=sr)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|", w=None):

    if w == 'train':
        file = 'transcript.txt'
    else:
        file = 'val.txt'

    filepaths_and_text = []
    for i, item in enumerate(filename):
        with open(item, encoding='utf-8') as f:
            for line in f:
                sp = line.split('|')
                filepaths_and_text.append([item.replace(file, '') + sp[0], sp[1], i])
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, w):
        _audiopaths_and_text = []
        for item in audiopaths_and_text:
            if w == 'train':
                _audiopaths_and_text.append(item + 'transcript.txt')
            else:
                _audiopaths_and_text.append(item + 'val.txt')

        self.audiopaths_and_text = load_filepaths_and_text(_audiopaths_and_text, w=w)
        self.max_wav_value = hparams['max_wav_value']
        self.sampling_rate = hparams['sample_rate']
        self.stft = layers.TacotronSTFT(
            hparams['filter_length'], hparams['hop_length'], hparams['win_length'],
            hparams['n_mel_channels'], hparams['sample_rate'], hparams['mel_fmin'],
            hparams['mel_fmax'])
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text, speaker_id = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel, speaker_id)

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename, self.stft.sampling_rate)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        speaker_id = []
        for i in range(len(batch)):
            speaker_id.append(batch[i][2])
        speaker_id = torch.FloatTensor(speaker_id)

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, speaker_id ,\
            output_lengths