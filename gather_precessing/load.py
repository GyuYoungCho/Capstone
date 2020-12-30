import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

def read_image(image_path):
    image=tf.io.read_file(image_path)
    return tf.image.decode_jpeg(image, channels=3)

def format(image, label):
    image = tf.cast(image,tf.float32)
    image = image/255
    return image, label

def load_data():
    with open('image96.txt', 'r') as f:
        filename = f.readlines()

    filename = [_.split('\n')[0] for _ in filename]
    label = pd.read_csv('label.csv', encoding='CP949')
    label = label[['강수15', 'fog', 'weather', 'sunny']]

    weatherlabel = {'맑음': 0, '안개': 1, '비': 2}
    foglabel = {'None': 0, 'Shallow': 1, 'Dense': 2}
    label['weather'] = label['weather'].map(weatherlabel)
    label['fog'] = label['fog'].map(foglabel)

    X_train, X_test, y_train, y_test = train_test_split(filename, label, test_size=0.2, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(100_000)

    train_dataset = train_dataset.map(lambda x, y: (read_image(x), y))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.map(format)

    valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    valid_dataset = valid_dataset.shuffle(100_000)
    valid_dataset = valid_dataset.map(lambda x, y: (read_image(x), y))
    valid_dataset = valid_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.map(format)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.shuffle(100_000)
    test_dataset = test_dataset.map(lambda x, y: (read_image(x), y))
    test_dataset = test_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(format)

    return train_dataset, valid_dataset, test_dataset

train_dataset, valid_dataset, test_dataset = load_data()

mb = tf.keras.applications.MobileNetV2(input_shape=(96,96,3),include_top=False)
mb.trainable = False

global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
last = tf.keras.layers.Dense(4)

model = tf.keras.models.Sequential([
    mb, global_average_pooling, last
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
             optimizer='adam',metrics=['acc'])

model.fit(train_dataset, steps_per_epoch=len(X_train)//256, validation_data = valid_dataset ,epochs=3)
