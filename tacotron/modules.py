# coding: utf-8
# Code based on https://github.com/keithito/tacotron/blob/master/models/tacotron.py

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tensorflow as tf
from tensorflow.keras.layers import GRUCell
from tensorflow.python.ops import init_ops
from tensorflow.python.layers import core


def get_embed(inputs, num_inputs, embed_size, name):  # speaker_id, self.num_speakers, hp.enc_prenet_sizes[-1], "before_highway"
    embed_table = tf.get_variable(name, [num_inputs, embed_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
    return tf.nn.embedding_lookup(embed_table, inputs)

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class CBHG(nn.Module):
    def __init__(self, in_dim, K=16, projections=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList([BatchNormConv1d(in_dim, in_dim, k, stride=1, padding='same', activation=self.relu)
                                      for k in range(1, K + 1)])
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding='same')

        in_sizes = [K * in_dim] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1) + [None]
        self.conv1d_projections = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1,
                             padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(
                in_sizes, projections, activations)])

        self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False)

        self.highways = nn.ModuleList([Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(
            in_dim, in_dim, 1, batch_first=True, bidirectional=True)

        def forward(self, inputs, input_lengths=None):
            # (B, T_in, in_dim)
            x = inputs

            # Needed to perform conv1d on time-axis
            # (B, in_dim, T_in)
            if x.size(-1) == self.in_dim:
                x = x.transpose(1, 2)

            T = x.size(-1)

            # (B, in_dim*K, T_in)
            # Concat conv1d bank outputs
            x = torch.cat([conv1d(x)[:, :, :T] for conv1d in self.conv1d_banks], dim=1)
            assert x.size(1) == self.in_dim * len(self.conv1d_banks)
            x = self.max_pool1d(x)[:, :, :T]

            for conv1d in self.conv1d_projections:
                x = conv1d(x)

            # (B, T_in, in_dim)
            # Back to the original shape
            x = x.transpose(1, 2)

            if x.size(-1) != self.in_dim:
                x = self.pre_highway(x)

            # Residual connection
            x += inputs
            for highway in self.highways:
                x = highway(x)

            if input_lengths is not None:
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True)

            # (B, T_in, in_dim*2)
            outputs, _ = self.gru(x)

            if input_lengths is not None:
                outputs, _ = nn.utils.rnn.pad_packed_sequence(
                    outputs, batch_first=True)

            return outputs

class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def     __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams['num_mels'], hparams['postnet_embedding_dim'],
                         kernel_size=hparams['postnet_kernel_size'], stride=1,
                         padding=int((hparams['postnet_kernel_size'] - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams['postnet_embedding_dim']))
        )

        for i in range(1, hparams['postnet_n_convolutions'] - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams['postnet_embedding_dim'],
                             hparams['postnet_embedding_dim'],
                             kernel_size=hparams['postnet_kernel_size'], stride=1,
                             padding=int((hparams['postnet_kernel_size'] - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams['postnet_embedding_dim']))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams['postnet_embedding_dim'], hparams['num_mels'],
                         kernel_size=hparams['postnet_kernel_size'], stride=1,
                         padding=int((hparams['postnet_kernel_size'] - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams['num_mels']))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x

class BatchNormConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding,
                 activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        # Following tensorflow's default parameters
        self.bn = nn.BatchNorm1d(out_dim, momentum=0.99, eps=1e-3)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)

class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


def batch_tile(tensor, batch_size):
    expaneded_tensor = tf.expand_dims(tensor, [0])
    return tf.tile(expaneded_tensor, \
            [batch_size] + [1 for _ in tensor.get_shape()])

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(BiLSTMModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.flstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.blstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, 2)
        self.embed.weight.data.uniform_(-1, 1)

        self.use_cuda = args.use_cuda

    def forward(self, x, mask, is_eval=False):
        '''
            run the model, take input sentence and predict the logits
            args:
                x: encoded sentence
                mask: the mask of the sentence
            returns:

        '''
        x_embd = self.embed(x)
        # forward lstm
        fout, (hn, cn) = self.flstm(x_embd)

        # calculate backward index
        rev_index = torch.range(x.size(1) - 1, 0, -1).view(1, -1).expand(x.size(0), x.size(1)).long()
        if self.use_cuda:
            rev_index = rev_index.cuda()
        # code.interact(local=locals())
        mask_length = torch.sum(1 - mask.data, 1).unsqueeze(1).long().expand_as(rev_index)
        rev_index -= mask_length
        rev_index[rev_index < 0] = 0
        rev_index = Variable(rev_index, volatile=is_eval)

        # reverse the order of x and store it in bx
        bx = Variable(x.data.new(x.size()).fill_(0), volatile=is_eval)
        bx = torch.gather(x, 1, rev_index)
        bx_embd = self.embed(bx)
        # backward lstm
        bout, (hn, cn) = self.blstm(bx_embd)

        # concat forward hidden states with backward hidden states
        out = torch.cat([fout, bout], 2)
        length = mask.sum(1).unsqueeze(1).unsqueeze(2).expand(out.size(0), 1, out.size(2)).long() - 1

        # gather the last hidden states
        out = torch.gather(out, 1, length).contiguous().squeeze(1)
        out = self.linear(out)
        return out

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding='same', dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

def get_mask_from_lengths(lengths):
    lengths = torch.from_numpy(lengths)
    max_len = torch.max(lengths).item()
    if torch.cuda.is_available():
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    else:
        ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask