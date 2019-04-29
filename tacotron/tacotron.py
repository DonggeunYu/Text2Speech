import tensorflow as tf
import tacotron.attention_wrapper as attention_wrapper
import torch.nn as nn
import torch
import numpy as np

from text.symbols import symbols
from utils.infolog import log
from .modules import *
from .helpers import TacoTestHelper, TacoTrainingHelper
from utils.core_rnn_cell import OutputProjectionWrapper
from utils.core_rnn_cell import GRUCell
from tensorflow.compat.v1.keras.layers import StackedRNNCells
from tensorflow.compat.v1.nn.rnn_cell import ResidualWrapper

class Tacotron(nn.Module):
    def __init__(self, hparams, inputs):
        super(Tacotron, self).__init__()
        self._hparams = hparams
        hp = self._hparams

        # Embeddings(256)
        self.char_embed_table = nn.Embedding(len(symbols), hp['embedding_size'])

        self.encoder = Encoder(hp)

class Encoder(nn.Module):
    def __init__(self, hp):
        super(Encoder, self).__init__()
        self.prenet = prenet(hp['embedding_size'], hp['enc_prenet_sizes'])
        self.cbhg = cbhg(128, K=16, projections=[128, 128])

    def forward(self, inputs, input_lengths=None):
        inputs = self.prenet(inputs)
        return self.cbhg(inputs, input_lengths)