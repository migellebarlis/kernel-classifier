import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

img = tf.squeeze(tf.image.convert_image_dtype(tf.io.decode_png(tf.io.read_file("Kernel8G.png")), dtype=tf.float32)).numpy()
# img = tf.squeeze(tf.image.rgb_to_grayscale(tf.image.convert_image_dtype(tf.io.decode_png(tf.io.read_file("Kernel1G.png")), dtype=tf.float32))).numpy()
fft = tf.signal.fftshift(tf.abs(tf.signal.fft2d(tf.cast(img, dtype=tf.complex64)))).numpy()

plt.figure()
plt.imshow(img)
plt.show()

plt.figure()
plt.imshow(fft)
plt.show()
