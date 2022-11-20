import glob
import os
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt

images = glob.glob('src/*/*/*.png')

for i in images:
    rgb = tf.keras.utils.load_img(i)
    ycbcr = tfio.experimental.color.rgb_to_ycbcr(rgb)
    y = ycbcr[:,:,0]
    path = i.split('\\')
    path[1] = path[1] + '_y'
    tf.keras.utils.save_img(os.path.join(*path),y)