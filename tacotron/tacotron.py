import tensorflow as tf

from text.symbols import symbols

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

            