import tensorflow as tf
from keras.layers import Layer


class FFT2(Layer):
    def __init__(self, **kwargs):
        super(FFT2, self).__init__(**kwargs)

    def call(self, input_data):
        return tf.signal.fftshift(tf.abs(tf.signal.fft2d(tf.cast(input_data, dtype=tf.complex64))))
