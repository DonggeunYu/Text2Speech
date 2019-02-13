import tensorflow as tf

def abc(a, b):
    print(a, b)
lin = 1
a = tf.py_func(abc, [[1, 2, 3], ['a', 'b', 'c']], lin)
