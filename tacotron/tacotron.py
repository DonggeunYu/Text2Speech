import tensorflow as tf

from text.symbols import symbols
from .modules import *

class Tacotron():
    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self, inputs, input_lengths, num_speakers, speaker_id, mel_targets=None, linear_targets=None, loss_coeff=None,
                   rnn_decoder_test_mode=False, is_randomly_initialized=False):
        is_training2 = linear_targets is not None
        is_training = not rnn_decoder_test_mode

        self.is_randomly_initialized = is_randomly_initialized

        with tf.variable_scope('inference') as scope:
            hp = self._hparams
            batch_size = tf.shape(inputs)[0]

            #Embeddings(256)
            char_embed_table = tf.get_variable('embedding', [len(symbols), hp.embedding_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.5))

            zero_pad = True

            if zero_pad:
                #<PAD> 0 은 embedding이 0으로 고정되고, train으로 변하지 않는다. 즉, 위의 get_variable에서 잡았던 변수의 첫번째 행(<PAD>)에 대응되는 것은 사용되지 않는 것이다)
                char_embed_table = tf.concat((tf.zeros(shape=[1, hp.embedding_size]), char_embed_table[1:, :]), 0)

            #[N, T_in, embedding_size]
            #https://m.blog.naver.com/wideeyed/221343328832
            char_embedded_inputs = tf.nn.embedding_lookup(char_embed_table, inputs)

            self.num_speakers = num_speakers
            if self.num_speakers > 1:
                if hp.speaker_embedding_size != 1:
                    speaker_embed_table = tf.get_variable('speaker_embedding',
                                                          [self.num_speakers, hp.speaker_embedding_size],
                                                          dtype=tf.float32,
                                                          initializer=tf.truncated_normal_initializer(stddev=0.5))
                    # [N, T_in, speaker_embedding_size]
                    speaker_embed = tf.nn.embedding_lookup(speaker_embed_table, speaker_id)

                    if hp.model_type == 'deepvoice':
                        if hp.speaker_embedding_size == 1:
                            before_highway = get_embed(speaker_id, self.num_speakers, hp.enc_prenet_sizes[-1],
                                                       "before_highway")  # 'enc_prenet_sizes': [f(256), f(128)]
                            encoder_rnn_init_state = get_embed(speaker_id, self.num_speakers, hp.enc_rnn_size * 2,
                                                               "encoder_rnn_init_state")

                            attention_rnn_init_state = get_embed(speaker_id, self.num_speakers, hp.attention_state_size,
                                                                 "attention_rnn_init_state")
                            decoder_rnn_init_stdates = [get_embed(speaker_id, self.num_speakers, hp.dec_rnn_size,
                                                                  "decoder_rnn_init_states{}".format(idx + 1))
                                                        for idx in range(hp.dec_layer_num)]
                        else:
                            deep_dense = lambda x, dim: tf.layers.dense(x, dim,
                                                                        activation=tf.nn.softsign)  # softsign: x / (abs(x) + 1)

                            before_highway = deep_dense(speaker_embed, hp.enc_prenet_sizes[-1])
                            encoder_rnn_init_state = deep_dense(speaker_embed, hp.enc_rnn_size * 2)

                            attention_rnn_init_state = deep_dense(speaker_embed, hp.attention_state_size)
                            decoder_rnn_init_states = [deep_dense(speaker_embed, hp.dec_rnn_size) for _ in
                                                       range(hp.dec_layer_num)]

                        speaker_embed = None  # deepvoice does not use speaker_embed directly
                    elif hp.model_type == 'simple':
                        # simple model은 speaker_embed를 DecoderPrenetWrapper,ConcatOutputAndAttentionWrapper에 각각 넣어서 concat하는 방식이다.
                        before_highway = None
                        encoder_rnn_init_state = None
                        attention_rnn_init_state = None
                        decoder_rnn_init_states = None
                    else:
                        raise Exception(" [!] Unkown multi-speaker model type: {}".format(hp.model_type))
                else:
                    #self.num_speakers=1
                    speaker_embed = None
                    before_highway = None
                    encoder_rnn_init_state = None  # bidirectional GRU의 init state
                    attention_rnn_init_state = None
                    decoder_rnn_init_states = None

                ##########
                #Encoder
                ##########

                # [N, T_in, enc_prenet_sizes[-1]]
                prenet_outputs = prenet(char_embedded_inputs, is_training, hp.enc_prenet_sizes, hp.dropout_prob,
                                        scope='prenet')  # 'enc_prenet_sizes': [f(256), f(128)],  dropout_prob = 0.5
                # ==> (N, T_in, 128)

                # enc_rnn_size = 128
                encoder_outputs = cbhg(prenet_outputs, input_lengths, is_training, hp.enc_bank_size,
                                       hp.enc_bank_channel_size,
                                       hp.enc_maxpool_width, hp.enc_highway_depth, hp.enc_rnn_size, hp.enc_proj_sizes,
                                       hp.enc_proj_width,
                                       scope="encoder_cbhg", before_highway=before_highway,
                                       encoder_rnn_init_state=encoder_rnn_init_state)
                print(encoder_outputs)