import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from FFT3 import FFT3

TRAINING_DIR = 'src/training'
training_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = training_datagen.flow_from_directory(
  TRAINING_DIR,
  target_size = (800,800),
  class_mode = 'categorical'
)

VALIDATION_DIR = 'src/validation'
validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_directory(
  VALIDATION_DIR,
  target_size = (800,800),
  class_mode = 'categorical'
)

TESTING_DIR = 'src/testing'
testing_datagen = ImageDataGenerator(rescale = 1./255)
testing_generator = testing_datagen.flow_from_directory(
  TESTING_DIR,
  target_size = (800,800),
  class_mode = 'categorical'
)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(
    filters = 64,
    kernel_size = (3,3),
    activation = None,
    input_shape = (800,800,3)
  ),
  FFT3(),
  tf.keras.layers.Conv2D(
    filters = 64,
    kernel_size = (3,3),
    activation = 'relu',
    input_shape = (800,800,3)
  ),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(
    filters = 64,
    kernel_size = (3,3),
    activation = 'relu'
  ),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(
    filters = 128,
    kernel_size = (3,3),
    activation = 'relu'
  ),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(
    filters = 128,
    kernel_size = (3,3),
    activation = 'relu'
  ),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(8, activation='softmax')
])

model.compile(optimizer = 'rmsprop',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(
  train_generator,
  epochs= 25,
  validation_data = validation_generator,
  verbose = 1)

model.save('model/kernel_classifier')

model.evaluate_generator(testing_generator)