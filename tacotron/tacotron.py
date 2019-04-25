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

class embedding(nn.Module):
    def __init__(self, hparams, inputs):
        self._hparams = hparams
        hp = self._hparams

        # Embeddings(256)
        char_embed_table = nn.Embedding(len(symbols), hp['embedding_size'])


        # [N, T_in, embedding_size]
        print(np.shape(inputs))
        for input in inputs:
            char_embedded_inputs = [char_embed_table[w] for w in input]

        return char_embedded_inputs

class Tacotron():
    def __init__(self, hparams, inputs):
        self._hparams = hparams
        hp = self._hparams

        # Embeddings(256)
        char_embed_table = nn.Embedding(len(symbols), hp['embedding_size'])

        zero_pad = False

        if zero_pad:
            # <PAD> 0 은 embedding이 0으로 고정되고, train으로 변하지 않는다. 즉, 위의 get_variable에서 잡았던 변수의 첫번째 행(<PAD>)에 대응되는 것은 사용되지 않는 것이다)
            char_embed_table = torch.cat([torch.zeros(1, hp['embedding_size']), char_embed_table[1:, :]], 0)

            # [N, T_in, embedding_size]
        print(np.shape(inputs))
        for input in inputs:
            char_embedded_inputs = [char_embed_table[w] for w in input]

        return char_embedded_inputs


    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.compat.v1.variable_scope('loss') as scope:
            hp = self._hparams
            mel_loss = tf.abs(self.mel_targets - self.mel_outputs)

            l1 = tf.abs(self.linear_targets - self.linear_outputs)
            expanded_loss_coeff = tf.expand_dims(tf.expand_dims(self.loss_coeff, [-1]), [-1])

            if hp.prioritize_loss:
                # Prioritize loss for frequencies.
                upper_priority_freq = int(5000 / (hp.sample_rate * 0.5) * hp.num_freq)
                lower_priority_freq = int(165 / (hp.sample_rate * 0.5) * hp.num_freq)

                l1_priority= l1[:,:,lower_priority_freq:upper_priority_freq]

                self.loss = tf.reduce_mean(mel_loss * expanded_loss_coeff) + \
                        0.5 * tf.reduce_mean(l1 * expanded_loss_coeff) + 0.5 * tf.reduce_mean(l1_priority * expanded_loss_coeff)
                self.linear_loss = tf.reduce_mean( 0.5 * (tf.reduce_mean(l1) + tf.reduce_mean(l1_priority)))
            else:
                self.loss = tf.reduce_mean(mel_loss * expanded_loss_coeff) + tf.reduce_mean(l1 * expanded_loss_coeff)   # optimize할 때는 self.loss를 사용하고, 출력할 때는 아래의 loss_without_coeff를 사용함
                self.linear_loss = tf.reduce_mean(l1)

            self.mel_loss = tf.reduce_mean(mel_loss)
            self.loss_without_coeff = self.mel_loss + self.linear_loss

    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
            global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            hp = self._hparams

            step = tf.cast(global_step + 1, dtype=tf.float32)

            if hp.decay_learning_rate_mode == 0:
                if self.is_randomly_initialized:
                    warmup_steps = 4000.0
                else:
                    warmup_steps = 40000.0
                self.learning_rate = hp.tacotron_initial_learning_rate * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
            elif hp.decay_learning_rate_mode == 1:
                self.learning_rate = hp.tacotron_initial_learning_rate * tf.train.exponential_decay(1., step, 3000, 0.95)

            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),global_step=global_step)

    def get_dummy_feed_dict(self):
        feed_dict = {self.is_manual_attention: False, self.manual_alignments: np.zeros([1, 1, 1]),}
        return feed_dict
