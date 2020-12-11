"Author: Sam Gronseth"
import glob2

import tensorflow as tf
from tensorflow import keras
import numpy as np
# import pydotplus
# import graphviz
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
    img_paths = list()
    # Recursively find all the image files from the path data_path
    for img_path in glob2.glob(data_path + "/**/*"):
        img_paths.append(img_path)
    images = np.zeros((len(img_paths), 256, 256))
    labels = np.zeros(len(img_paths))

    # Read and resize the images
    # Get the encoded labels
    for i, img_path in enumerate(img_paths):
        images[i] = get_pic(img_path)
        labels[i] = label_to_index[get_label(img_path)]

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255, validation_split=0.2)

    train_images = image_generator.flow_from_directory(batch_size=32,
                                                        directory='Lung_imgs',
                                                        shuffle=True,
                                                        target_size=(280, 280),
                                                        subset="training",
                                                        class_mode='categorical')

    validation_images = image_generator.flow_from_directory(batch_size=32,
                                                             directory='Lung_imgs',
                                                             shuffle=True,
                                                             target_size=(280, 280),
                                                             subset="validation",
                                                             class_mode='categorical')

    if not training:
        return validation_images, validation_images.class_indices
    else:
        return train_images, train_images.class_indices


def build_model():                                                                                       # model = build_model()
    model = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
        keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_model(model, train_img, train_lab, test_img, test_lab, T):                                # train_model(model, train_images, train_images.class_indices, test_images, test_images.class_indices, 1)
    train_lab = keras.utils.to_categorical(train_lab)
    test_lab = keras.utils.to_categorical(test_lab)

    model.fit(train_img, train_lab, validation_data=(test_img, test_lab), epochs=T)


def predict_label(model, images, index):                                             # predict_label(model, test_images, 0)
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
# test_images, test_labels = get_ds("/Lung_imgs/")
#
# # Finally train it
# model.fit(train_X,train_y, validation_data=(val_X,val_y))
#
# # Predictions
# model.predict(val_X)