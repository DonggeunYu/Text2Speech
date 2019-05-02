import torch.nn as nn
import torch
import numpy as np

from text.symbols import symbols
from utils.infolog import log
from .modules import *
from .attention import get_mask_from_lengths, BahdanauAttention, AttentionWrapper
from torch.autograd import Variable
from hparams import hparams


class Tacotron(nn.Module):
    def __init__(self, hparams, n_vocab, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False, num_speakers=1):
        super(Tacotron, self).__init__()
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.use_memory_mask = use_memory_mask
        embedding_dim = hparams['embedding_size']
        self.char_embed_table = nn.Embedding(n_vocab, embedding_dim)
        std = np.sqrt(2.0 / (n_vocab + embedding_dim))
        val = np.sqrt(3.0) * std  # uniform bounds for std
        self.char_embed_table.weight.data.uniform_(-val, val)

        if num_speakers > 1:
            self.speaker_embed_table = nn.Embedding(num_speakers, hparams['speaker_embedding_size'])
            self.deep_dense = lambda x, dim: nn.Sequential(nn.Linear(x, dim), nn.Softsign())  # softsign: x / (abs(x) + 1)

        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(mel_dim, r)

        self.postnet = CBHG(mel_dim, K=8, projections=[256, mel_dim])
        self.last_linear = nn.Linear(mel_dim * 2, linear_dim)

    def forward(self, num_speakers, inputs, input_lengths, loss_coeff=None, mel_targets=None, linear_targets=None, stop_token_targets=None, speaker_id=None):
        self.num_speakers = num_speakers

        B = inputs.size(0)

        char_embedded_inputs = self.char_embed_table(inputs).view((1, -1))

        if num_speakers > 1:
            speaker_embed = self.speaker_embed_table(speaker_id).view((1, -1))

            encoder_rnn_init_state = self.deep_dense(np.shape(speaker_embed)[1],
                                                hparams['encoder_lstm_units'] * 4)  # hp.encoder_lstm_units = 256


            decoder_rnn_init_states = [
                self.deep_dense(np.shape(speaker_embed)[1], self.decoder_lstm_units * 2) for i in
                range(hparams['dec_layer_num'])]  # hp.decoder_lstm_units = 1024

            speaker_embed = None
        else:
            # self.num_speakers =1인 경우
            speaker_embed = None
            encoder_rnn_init_state = None  # bidirectional GRU의 init state
            attention_rnn_init_state = None
            decoder_rnn_init_states = None

        # (B, T', in_dim)
        encoder_outputs = self.encoder(inputs, input_lengths,
                                       encoder_rnn_init_state, attention_rnn_init_state, decoder_rnn_init_states)

        if self.use_memory_mask:
            memory_lengths = input_lengths
        else:
            memory_lengths = None
        # (B, T', mel_dim*r)
        mel_outputs, alignments = self.decoder(
            encoder_outputs, input_lengths, memory_lengths=memory_lengths)

        # Post net processing below

        # Reshape
        # (B, T, mel_dim)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments

class Encoder(nn.Module):
    def __init__(self, hp):
        super(Encoder, self).__init__()
        hp = hparams
        self.conv1d = nn.Conv1d(in_channels=hp['enc_conv_channels'], out_channels=hp['enc_conv_channels'], kernel_size=hp['enc_conv_kernel_size'], padding='same')
        self.conv1d = nn.ReLU(self.conv1d)
        self.prenet = Prenet(hp['embedding_size'], hp['enc_prenet_sizes'])
        self.cbhg = CBHG(128, K=16, projections=[128, 128])

    def forward(self, x, input_lengths=None, encoder_rnn_init_state=None, attention_rnn_init_state=None, decoder_rnn_init_states=None):

        for i in range(hparams['enc_conv_num_layers']):
            x = self.conv1d(x)
            x = nn.BatchNorm2d(x)
            x = nn.Dropout(x)

        if encoder_rnn_init_state is not None:
            initial_state_fw_c, initial_state_fw_h, initial_state_bw_c, initial_state_bw_h = tf.split(
                encoder_rnn_init_state, 4, 1)
            initial_sate_fw = BiLSTMModel(initial_state_bw_c,initial_state_bw_h)
            initial_state_bw = BiLSTMModel(initial_state_bw_c,initial_state_bw_h)



class Decoder(nn.Module):
    def __init__(self, in_dim, r):
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.r = r
        self.prenet = Prenet(in_dim * r, sizes=[256, 128])
        # (prenet_out + attention context) -> output
        self.attention_rnn = AttentionWrapper(
            nn.GRUCell(256 + 128, 256),
            BahdanauAttention(256)
        )
        self.memory_layer = nn.Linear(256, 256, bias=False)
        self.project_to_decoder_in = nn.Linear(512, 256)

        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(256, 256) for _ in range(2)])

        self.proj_to_mel = nn.Linear(256, in_dim * r)
        self.max_decoder_steps = 200

    def forward(self, encoder_outputs, inputs=None, memory_lengths=None):
        """
        Decoder forward step.
        If decoder inputs are not given (e.g., at testing time), as noted in
        Tacotron paper, greedy decoding is adapted.
        Args:
            encoder_outputs: Encoder outputs. (B, T_encoder, dim)
            inputs: Decoder inputs. i.e., mel-spectrogram. If None (at eval-time),
              decoder outputs are used as decoder inputs.
            memory_lengths: Encoder output (memory) lengths. If not None, used for
              attention masking.
        """
        B = encoder_outputs.size(0)

        processed_memory = self.memory_layer(encoder_outputs)
        if memory_lengths is not None:
            mask = get_mask_from_lengths(processed_memory, memory_lengths)
        else:
            mask = None

        # Run greedy decoding if inputs is None
        greedy = inputs is None

        if inputs is not None:
            # Grouping multiple frames if necessary
            if inputs.size(-1) == self.in_dim:
                inputs = inputs.view(B, inputs.size(1) // self.r, -1)
            assert inputs.size(-1) == self.in_dim * self.r
            T_decoder = inputs.size(1)

        # go frames
        initial_input = Variable(
            encoder_outputs.data.new(B, self.in_dim * self.r).zero_())

        decoder_init_state = list(initial_input)

        for idx, cell in enumerate(initial_input):
            shape1 = decoder_init_state[idx][0].get_shape().as_list()
            shape2 = cell.get_shape().as_list()
            if shape1[1] * 2 != shape2[1]:
                raise Exception(" [!] Shape {} and {} should be equal".format(shape1, shape2))
            c, h = tf.split(cell, 2, 1)
            decoder_init_state[idx] = tuple(c, h)

        decoder_init_state = tuple(decoder_init_state)

            # Init decoder states
        attention_rnn_hidden = Variable(
            encoder_outputs.data.new(B, 256).zero_())
        decoder_rnn_hiddens = [Variable(
            encoder_outputs.data.new(B, 256).zero_())
            for _ in range(len(self.decoder_rnns))]
        current_attention = Variable(
            encoder_outputs.data.new(B, 256).zero_())

        # Time first (T_decoder, B, in_dim)
        if inputs is not None:
            inputs = inputs.transpose(0, 1)

        outputs = []
        alignments = []

        t = 0
        current_input = initial_input
        while True:
            if t > 0:
                current_input = outputs[-1] if greedy else inputs[t - 1]
            # Prenet
            current_input = self.prenet(current_input)

            # Attention RNN
            attention_rnn_hidden, current_attention, alignment = self.attention_rnn(
                current_input, current_attention, attention_rnn_hidden,
                encoder_outputs, processed_memory=processed_memory, mask=mask)

            # Concat RNN output and attention context vector
            decoder_input = self.project_to_decoder_in(
                torch.cat((attention_rnn_hidden, current_attention), -1))

            # Pass through the decoder RNNs
            for idx in range(len(self.decoder_rnns)):
                decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                    decoder_input, decoder_rnn_hiddens[idx])
                # Residual connectinon
                decoder_input = decoder_rnn_hiddens[idx] + decoder_input

            output = decoder_input
            output = self.proj_to_mel(output)

            outputs += [output]
            alignments += [alignment]

            t += 1

            if greedy:
                if t > 1 and is_end_of_frames(output):
                    break
                elif t > self.max_decoder_steps:
                    print("Warning! doesn't seems to be converged")
                    break
            else:
                if t >= T_decoder:
                    break

        assert greedy or len(outputs) == T_decoder

        # Back to batch first
        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        return outputs, alignments


def is_end_of_frames(output, eps=0.2):
    return (output.data <= eps).all()
