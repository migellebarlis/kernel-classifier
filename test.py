import tensorflow as tf
from FFT2 import FFT2


def parse_image(filename):
    # Remove the file extension
    file = tf.strings.split(filename, '.')[-2]

    # Get the zero-based kernel number
    kernel = tf.strings.to_number(tf.strings.split(file, '_')[-1]) - 1

    # Load the image
    image = tf.io.read_file(filename)
    image = tf.io.decode_png(image)

    return image, kernel


list_ds = tf.data.Dataset.list_files(str('test/*.png'), shuffle=True)
test_ds = list_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(100)
test_ds = test_ds.prefetch(1)

model = tf.keras.models.load_model('kernel_net', custom_objects={'FFT2': FFT2})

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_ds, batch_size=100)
print("test loss:\t\t\t\t\t{0}\ntest mean absolute error:\t\t\t{1}\ntest mean absolute percentage error:\t\t{2}".format(results[0], results[1], results[2]))

# Testing a single image
# print(tf.math.argmax(model.predict(tf.reshape(parse_image('test/065_008.png')[0], (1,128,128,1))), axis=1).numpy())