import tensorflow as tf
initial_cell_state = [[[1, 2, 3], [4, 5, 6]],[[1, 2, 3], [4, 5, 6]]]
tf.keras.layers.Flatten(initial_cell_state)