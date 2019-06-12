import torch
from math import sqrt
from .modules import *
from .attention import BahdanauAttention, AttentionWrapper
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

        self.embedding = nn.Embedding(n_vocab, embedding_dim)
        std = sqrt(2.0 / (n_vocab + embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        self.speaker_embed_table = nn.Embedding(num_speakers, hparams['speaker_embedding_size'])
        self.deep_dense = lambda x, dim: nn.Sequential(nn.Linear(x, dim), nn.Softsign())  # softsign: x / (abs(x) + 1)
        self.deep_linear = nn.Linear(hparams['speaker_embedding_size'], 512)

        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(mel_dim, r)

        self.postnet = Postnet(hparams)
        self.postnet_linear = nn.Linear(80, 1024)

    def forward(self, num_speakers, inputs, input_lengths, loss_coeff, mel_targets=None, linear_targets=None, stop_token_targets=None, speaker_id=None):
        self.num_speakers = num_speakers.size(0)
        # (B, T', in_dim)
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        embedded_inputs = embedded_inputs.contiguous()
        B_size = embedded_inputs.size(0)
        embedded_inputs = embedded_inputs.view(B_size, -1)

        speaker_embed = self.speaker_embed_table(speaker_id)
        speaker_embed = self.deep_linear(speaker_embed)
        speaker_embed = speaker_embed.view(speaker_embed.size(0), -1)
        embed = torch.cat([embedded_inputs, speaker_embed], 1).view(B_size, 512, -1)

        encoder_outputs = self.encoder(embed, input_lengths)

        # (B, T', mel_dim*r)
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_targets, memory_lengths=input_lengths)

        # Post net processing below
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        B_size = mel_outputs_postnet.size(0)
        mel_outputs_postnet = mel_outputs_postnet.view(B_size, -1, 80)

        #mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        #mel_outputs_postnet = self.postnet_linear(mel_outputs_postnet)

        #mel_outputs = mel_outputs.contiguous()
        #B_size = mel_outputs.size(0)
        #mel_outputs = mel_outputs.view(B_size, -1, 80)

        return [mel_outputs_postnet, gate_outputs, alignments]

        #return [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]

    def inference(self, inputs, speaker_id):
        # (B, T', in_dim)
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        embedded_inputs = embedded_inputs.contiguous()
        B_size = embedded_inputs.size(0)
        embedded_inputs = embedded_inputs.view(B_size, -1)

        speaker_embed = self.speaker_embed_table(speaker_id)
        speaker_embed = self.deep_linear(speaker_embed)
        speaker_embed = speaker_embed.view(speaker_embed.size(0), -1)
        embed = torch.cat([embedded_inputs, speaker_embed], 1).view(B_size, 512, -1)

        encoder_outputs = self.encoder.inference(embed)

        # (B, T', mel_dim*r)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        # Post net processing below
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        B_size = mel_outputs_postnet.size(0)
        mel_outputs_postnet = mel_outputs_postnet.view(B_size, -1, 80)

        #mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        #mel_outputs_postnet = self.postnet_linear(mel_outputs_postnet)

        #mel_outputs = mel_outputs.contiguous()
        #B_size = mel_outputs.size(0)
        #mel_outputs = mel_outputs.view(B_size, -1, 80)

        return [mel_outputs_postnet, gate_outputs, alignments]

        #return [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights

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
                nn.BatchNorm1d(hp['enc_conv_channels']),
                nn.Dropout())
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hp['embedding_size'],
                            int(hp['embedding_size'] / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths=None):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu().numpy()
        # pytorch tensor are not reversible, hence the conversion
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)


        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)


        return outputs
class Decoder(nn.Module):
    def __init__(self, in_dim, r):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams['num_mels']
        self.n_frames_per_step = hparams['n_frames_per_step']
        self.encoder_embedding_dim = hparams['embedding_size']
        self.attention_rnn_dim = hparams['attention_rnn_dim']
        self.decoder_rnn_dim = hparams['decoder_rnn_dim']
        self.prenet_dim = hparams['prenet_dim']
        self.max_decoder_steps = hparams['max_decoder_steps']
        self.gate_threshold = hparams['gate_threshold']
        self.p_attention_dropout = hparams['p_attention_dropout']
        self.p_decoder_dropout = hparams['p_decoder_dropout']

        self.in_dim = in_dim
        self.r = r
        self.prenet = Prenet(
            hparams['num_mels'] * hparams['n_frames_per_step'],
            [hparams['prenet_dim'], hparams['prenet_dim']])
        # (prenet_out + attention context) -> output
        self.attention_rnn = AttentionWrapper(
            nn.GRUCell(256 + 128, 256),
            BahdanauAttention(256)
        )
        #self.memory_layer = nn.Linear(256, 512, bias=False)
        #self.project_to_decoder_in = nn.Linear(512, 256)

        #self.decoder_rnns = nn.ModuleList(
            #[nn.GRUCell(256, 256) for _ in range(2)])

        self.attention_rnn = nn.LSTMCell(
            hparams['prenet_dim'] + hparams['embedding_size'],
            hparams['attention_rnn_dim'])

        self.attention_layer = Attention(
            hparams['attention_rnn_dim'], hparams['embedding_size'],
            hparams['attention_dim'], hparams['attention_location_n_filters'],
            hparams['attention_location_kernel_size'])

        self.decoder_rnn = nn.LSTMCell(
            hparams['attention_rnn_dim'] + hparams['embedding_size'],
            hparams['decoder_rnn_dim'], 1)

        self.linear_projection = LinearNorm(
            hparams['decoder_rnn_dim'] + hparams['embedding_size'],
            hparams['num_mels'] * hparams['n_frames_per_step'])

        self.gate_layer = LinearNorm(
            hparams['decoder_rnn_dim'] + hparams['embedding_size'], 1,
            bias=True, w_init_gain='sigmoid')

        self.proj_to_mel = nn.Linear(256, in_dim * r)

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
        # input shape: (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output
        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training)
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(
            self.decoder_cell, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

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

        decoder_input = self.get_go_frame(encoder_outputs).unsqueeze(0)
        inputs = self.parse_decoder_inputs(inputs)
        inputs = torch.cat((decoder_input, inputs), dim=0)
        inputs = self.prenet(inputs)

        self.initialize_decoder_states(
            encoder_outputs, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < inputs.size(0) - 1:
            decoder_input = inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments



def is_end_of_frames(output, eps=0.2):
    return (output.data <= eps).all()
