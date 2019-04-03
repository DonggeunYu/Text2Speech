from __future__ import print_function

import tensorflow as tf

import time
tf.compat.v2.disable_eager_execution()

_placeholders = [
            tf.keras.backend.placeholder([1, 1], dtype=tf.int32, name='inputs'),
            tf.keras.backend.placeholder([1], dtype=tf.int32, name='input_lengths'),
            tf.keras.backend.placeholder([1], dtype=tf.float32, name='loss_coeff'),
            tf.keras.backend.placeholder([1, 1,10], dtype=tf.float32, name='mel_targets'),
            tf.keras.backend.placeholder([1, 1, 10], dtype=tf.float32, name='linear_targets'),
        ]
dtypes = [tf.int32, tf.int32, tf.float32, tf.float32, tf.float32]

queue = tf.queue.FIFOQueue(capacity=10, dtypes=dtypes)
enque = queue.enqueue(_placeholders)
