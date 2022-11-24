from random import shuffle
import tensorflow as tf
import numpy as np
import cv2
import scipy as sp
import pickle
from FFT2 import FFT2


def parse_image(filename):
    # Initialize patch size
    n = 128
    nn = n*n

    print(filename)

    # Remove the file extension
    file = tf.strings.split(filename, '.')[-2]

    # Get the kernel number
    kernel = tf.strings.to_number(tf.strings.split(file, '_')[-1])

    # Load the image
    image = tf.io.read_file(filename)
    image = tf.io.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # if image.shape[2] > 1:
    #     image = tf.image.rgb_to_grayscale(image)
    image = tf.squeeze(image)
    # image = cv2.imread(filename)

    # image = tf.keras.utils.load_img(filename)

    # Covert to grayscale
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = tf.image.rgb_to_grayscale(image)

    # Convert to float32
    # image = image/255

    # Get patch with the highest variance
    im2 = image*image
    im2conv = sp.signal.convolve2d(in1=im2, in2=np.ones((n, n)), mode='valid')
    imconv = sp.signal.convolve2d(in1=image, in2=np.ones((n, n)), mode='valid')
    imconv2 = imconv*imconv

    variance = (im2conv - (imconv2 / nn)) / nn
    max = np.argmax(variance)
    max = np.unravel_index(max, variance.shape)

    patch = image[max[0]:(max[0]+n), max[1]:(max[1]+n)]

    # Extract patches
    # patches = tf.image.extract_patches(images=tf.reshape(image, (1, tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2])),
    #                          sizes=[1, n, n, 1],
    #                          strides=[1, 1, 1, 1],
    #                          rates=[1, 1, 1, 1],
    #                          padding='VALID')

    # Get variances
    # variances = tf.math.reduce_variance(patches, axis=3)

    # Determine the highest variances
    # max2 = tf.math.reduce_max(variances, axis=2)
    # max2_idx = tf.math.argmax(variances, axis=2)
    # max1_idx = tf.math.argmax(max2, axis=1)

    # patch = patches[0, max1_idx[0], max2_idx[0, max1_idx[0]], :]
    # patch = tf.reshape(patch, (n, n, 1))

    # Get patch with the highest variance
    # im2 = cv2.multiply(image, image)
    # im2fil = cv2.filter2D(src=im2, ddepth=-1, kernel=np.ones((n, n)), borderType=cv2.BORDER_REPLICATE)

    # imfil = cv2.filter2D(src=image, ddepth=-1, kernel=np.ones((n, n)), borderType=cv2.BORDER_REPLICATE)
    # imfil2 = cv2.multiply(imfil, imfil)

    # variance = (im2fil - (imfil2 / nn)) / nn
    # max = np.argmax(variance)
    # max = np.unravel_index(max, variance.shape)

    # patch = image[max[0]:(max[0]+n), max[1]:(max[1]+n)]

    return patch, kernel


# test = parse_image('src/training/013/001_001.png')

list_ds = tf.data.Dataset.list_files(str('src/training/*/*.png'), shuffle=True)
train_ds = list_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(100)
train_ds = train_ds.prefetch(1)

list_ds = tf.data.Dataset.list_files(str('src/validation/*/*.png'), shuffle=True)
valid_ds = list_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.batch(100)
valid_ds = valid_ds.prefetch(1)

# Define the shared CNN model
shared_cnn = tf.keras.Sequential()
shared_cnn.add(tf.keras.layers.Input(shape=(128, 128, 1), name='cnn_input'))
shared_cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=None, padding='valid'))
shared_cnn.add(FFT2())
shared_cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid'))
shared_cnn.add(tf.keras.layers.MaxPool2D())
shared_cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid'))
shared_cnn.add(tf.keras.layers.MaxPool2D())
shared_cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
shared_cnn.add(tf.keras.layers.MaxPool2D())
shared_cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
shared_cnn.add(tf.keras.layers.MaxPool2D())
shared_cnn.add(tf.keras.layers.Flatten())
shared_cnn.add(tf.keras.layers.Dropout(0.5))
shared_cnn.add(tf.keras.layers.Dense(512, activation='relu'))
shared_cnn.add(tf.keras.layers.Dense(8, activation='softmax'))

# Compile the model and perform the training
model = tf.keras.models.Model(shared_cnn)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['MeanAbsoluteError', 'MeanAbsolutePercentageError'])

print(model.summary())
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='kernel_net', monitor='val_loss', save_best_only=True)
history = model.fit(x=train_ds, batch_size=100, epochs=500, validation_data=valid_ds, callbacks=[early_stop_callback, model_checkpoint_callback])

model.save('kernel_net')
with open('train.history', 'wb') as f:
    pickle.dump(history, f)

t = 0
