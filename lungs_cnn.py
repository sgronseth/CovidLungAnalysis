"Author: Sam Gronseth"

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from PIL import Image
import os


# TODO: main method to run this program (right now have to run functions individually)


# Respitory illness labels
labels_array = ['Pneumonia', 'Covid-19', 'Healthy', 'Large Cell Carcinoma', 'Adenocarcinoma']

label_to_index = dict((name, index) for index, name in enumerate(labels_array))


def get_pic(img_path):
    return np.array(Image.open(img_path).resize((256, 256), Image.ANTIALIAS))


def get_label(img_path):
    return Path(img_path).absolute().name


def get_ds(data_path, training=True):
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                      rotation_range=40,
                                                                      width_shift_range=0.2,
                                                                      shear_range=0.2,
                                                                      zoom_range=0.2,
                                                                      horizontal_flip=True)

    val_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_images = train_generator.flow_from_directory(batch_size=32,
                                                       directory='Lung_imgs/train',
                                                       shuffle=True,
                                                       target_size=(256, 256),
                                                       subset="training",
                                                       class_mode='categorical')

    validation_images = val_generator.flow_from_directory(batch_size=32,
                                                            directory='Lung_imgs',
                                                            shuffle=True,
                                                            target_size=(256, 256),
                                                            subset="validation",
                                                            class_mode='categorical')

    test_images = val_generator.flow_from_directory(batch_size=32,
                                                          directory='Lung_imgs/test',
                                                          shuffle=True,
                                                          target_size=(256, 256),
                                                          subset="validation",
                                                          class_mode='categorical')

    if not training:
        return test_images, test_images.class_indices
    else:
        return train_images, train_images.class_indices


def build_model():
    model = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(280, 280, 3)),
        keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_model(model, train_img, test_img, T):
    model.fit(train_img, validation_data=test_img, epochs=T)


def predict_label(model, images, index):
    prediction = model.predict(images)[index]
    print(prediction)

    pred_list = []
    i = 0
    for est in prediction:
        pred_list.append((labels_array[i], est))
        i += 1

    all_predictions = sorted(pred_list, key=lambda t: t[1], reverse=True)[:3]
    for result in all_predictions:
        print(result[0], ': ', "{:.2%}".format(result[1]), sep='')


# Load the train and validation data
# train_images, train_labels = get_ds("/Lung_imgs/")
# test_images, test_labels = get_ds("/Lung_imgs/", False)
#
# Build the model
# model = build_model()
#
# Train model
# train_model(model, train_images, test_images, 1)
#
# Predict results
# model.predict(val_X)
