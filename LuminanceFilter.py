import tensorflow as tf
import tensorflow_io as tfio
from keras.layers import Layer

class LuminanceFilter(Layer):
    def __init__(self, **kwargs):
        super(LuminanceFilter, self).__init__(**kwargs)
   
    def call(self, input_data):
        ycbcr = tfio.experimental.color.rgb_to_ycbcr(input_data)
        y = ycbcr[:,:,0]
        return y