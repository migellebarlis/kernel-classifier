from random import shuffle
import tensorflow as tf
import pickle
from FFT2 import FFT2


def parse_image(filename):
    # Remove the file extension
    file = tf.strings.split(filename, '.')[-2]

    # Get the kernel number
    kernel = tf.strings.to_number(tf.strings.split(file, '_')[-1])

    # Load the image
    image = tf.io.read_file(filename)
    image = tf.io.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.squeeze(image)

    return image, kernel


# test = parse_image('train/001_001.png')

list_ds = tf.data.Dataset.list_files(str('train/*.png'), shuffle=True)
train_ds = list_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(100)
train_ds = train_ds.prefetch(1)

list_ds = tf.data.Dataset.list_files(str('valid/*.png'), shuffle=True)
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
