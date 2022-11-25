import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from FFT2 import FFT2

with tf.keras.utils.custom_object_scope({'FFT2': FFT2}):
    with open('train.history','rb') as f:
        history = pickle.load(f)

print(history.history.keys())

plt.plot(history.history['mean_absolute_percentage_error'])
plt.plot(history.history['val_mean_absolute_percentage_error'])
plt.title('model error')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()