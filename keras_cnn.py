"Author: Sam Gronseth"

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pydotplus
import graphviz

class_names = [ 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot' ]


def get_dataset(training=True):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    if not training:
        return test_images, test_labels
    else:
        return train_images, train_labels


def build_model():  # model = build_model()
    model = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
        keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_model(model, train_img, train_lab, test_img, test_lab, T):    # train_model(model, train_images, train_labels, test_images, test_labels, 5)
    train_lab = keras.utils.to_categorical(train_lab)
    test_lab = keras.utils.to_categorical(test_lab)

    model.fit(train_img, train_lab, validation_data=(test_img, test_lab), epochs=T)


def predict_label(model, images, index):    # predict_label(model, test_images, 0)
    prediction = model.predict(images)[index]
    print(prediction)

    pred_list = [ ]
    i = 0
    for est in prediction:
        pred_list.append((class_names[ i ], est))
        i += 1

    top_three = sorted(pred_list, key=lambda t: t[ 1 ], reverse=True)[ :3 ]
    for pair in top_three:
        print(pair[ 0 ], ': ', "{:.2%}".format(pair[ 1 ]), sep='')