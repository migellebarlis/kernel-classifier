from typing import Any

import tensorflow as tf
import tensorflow_io as tfio
from keras.layers import Layer
from tensorflow import Tensor, IndexedSlices, SparseTensor


class LuminanceFilter(Layer):
    def __init__(self, **kwargs):
        super(LuminanceFilter, self).__init__(**kwargs)

    def call(self, input_data):
        ycbcr = tfio.experimental.color.rgb_to_ycbcr(tf.cast(input_data, dtype=tf.uint8))
        y = ycbcr[:, :, 0]
        return y


if __name__ == '__main__':
    print()
