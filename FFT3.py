import tensorflow as tf
from keras.layers import Layer

class FFT3(Layer):
    def __init__(self, **kwargs):
        super(FFT3, self).__init__(**kwargs)
   
    def call(self, input_data):
        return tf.signal.fftshift(tf.abs(tf.signal.fft3d(tf.cast(input_data, dtype=tf.complex64))))