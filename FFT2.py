import tensorflow as tf
from keras.layers import Layer


class FFT2(Layer):
    def __init__(self, **kwargs):
        super(FFT2, self).__init__(**kwargs)

    def call(self, input_data):
        permute = tf.transpose(input_data, perm=[0, 3, 1, 2])
        fft = tf.signal.fftshift(tf.abs(tf.signal.fft2d(tf.cast(permute, dtype=tf.complex64))))
        permute = tf.transpose(fft, perm=[0, 2, 3, 1])
        return permute
