import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np

from text.symbols import symbols
from utils.infolog import log
from .modules import *
from .attention import get_mask_from_lengths, BahdanauAttention, AttentionWrapper
from torch.autograd import Variable
from hparams import hparams

manualSeed = 999
torch.manual_seed(manualSeed)


class Tacotron(nn.Module):
    def __init__(self, hparams, n_vocab, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, num_speakers=1):
        super(Tacotron, self).__init__()
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        embedding_dim = hparams['embedding_size']
        self.char_embed_table = Variable(torch.zeros(n_vocab, embedding_dim), requires_grad=True)
        self.char_embed_table[:] = 0.5
        self.char_embed_table = torch.cat((Variable(torch.zeros(1, embedding_dim), ), self.char_embed_table[1:, :]), 0)

        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(mel_dim, r)

        self.postnet = CBHG(mel_dim, K=8, projections=[256, mel_dim])
        self.last_linear = nn.Linear(mel_dim * 2, linear_dim)

    def forward(self, num_speakers, inputs, input_lengths, loss_coeff=None, mel_targets=None, linear_targets=None, stop_token_targets=None, speaker_id=None):
        self.num_speakers = num_speakers

        B = inputs.size(0)
        char_embedded_inputs = F.embedding(inputs, self.char_embed_table)

        # (B, T', in_dim)
        char_embedded_inputs = char_embedded_inputs.transpose(1, 2)
        encoder_outputs = self.encoder(char_embedded_inputs, input_lengths)


        # (B, T', mel_dim*r)
        mel_outputs, alignments = self.decoder(
            encoder_outputs, mel_targets, memory_lengths=input_lengths)

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

        convolutions = []
        for _ in range(hp['enc_conv_num_layers']):
            conv_layer = nn.Sequential(
                ConvNorm(hp['enc_conv_channels'],
                         hp['enc_conv_channels'],
                         kernel_size=hp['enc_conv_kernel_size'], stride=1,
                         padding=int((hp['enc_conv_kernel_size'] - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hp['enc_conv_channels']))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hp['embedding_size'],
                            int(hp['embedding_size'] / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths=None):

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)


        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

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
        self.memory_layer = nn.Linear(256, 512, bias=False)
        self.project_to_decoder_in = nn.Linear(512, 256)

        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(256, 256) for _ in range(2)])

        self.proj_to_mel = nn.Linear(256, in_dim * r)
        self.max_decoder_steps = 200

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs
        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        RETURNS
        -------
        inputs: processed decoder inputs
        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

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

        decoder_input = self.get_go_frame(inputs).unsqueeze(0)
        inputs = self.parse_decoder_inputs(inputs)
        inputs = torch.cat((decoder_input, inputs), dim=0)
        inputs = self.prenet(inputs)

        self.initialize_decoder_states(
            encoder_outputs, mask=~get_mask_from_lengths(memory_lengths))

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
