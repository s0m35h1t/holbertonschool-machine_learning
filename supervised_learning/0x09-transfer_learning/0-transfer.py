#!/usr/bin/env python3
"""trains a convolutional neural network to classify the CIFAR 10 dataset"""

import tensorflow.keras as keras
import numpy as np


def preprocess_data(X, Y):
    """Pre-processes the data for your model:
    Args:
        X (numpy.ndarray) of shape (m, 32, 32, 3) containing the CIFAR 10
        data, where m is the number of data points
        Y (numpy.ndarray) of shape (m,) containing the CIFAR 10 labels
        for X
    Returns:
        X_p (numpy.ndarray) containing the preprocessed X
        Y_p (numpy.ndarray) containing the preprocessed Y"""

    X_p = Keras.applications.densenet.preprocess_input(X)
    Y_p = Keras.utils.to_categorical(Y, num_classes=10)

    return (X_p, Y_p)


if __name__ == "__main__":
    """transfer learning and trains a convolutional
    neural network to classify the CIFAR 10 datasetg"""

    model_name = "cifar10.h5"
    epochs = 40
    batch_size = 100
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = Keras.datasets.cifar10.load_data()
    y_test = Keras.utils.to_categorical(y_test, num_classes=10)
    y_train = Keras.utils.to_categorical(y_train, num_classes=10)

    data_augmentation_x = np.fliplr(x_train)
    data_augmentation_y = np.fliplr(y_train)
    x_train = np.concatenate([x_train, data_augmentation_x])
    y_train = np.concatenate([y_train, data_augmentation_y])

    x_train = Keras.applications.densenet.preprocess_input(x_train)
    x_test = Keras.applications.densenet.preprocess_input(x_test)

    pre_trained_model = Keras.applications.densenet.DenseNet201(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(32, 32, 3),
        pooling="avg",
        classes=num_classes)

    pre_trained_model.trainable = True
    ts = False
    for l in pre_trained_model.layers:
        if "conv5" in l.name or "conv4" in l.name:
            ts = True
        if ts:
            l.trainable = True
        else:
            l.trainable = False

    model = Keras.models.Sequential([
        pre_trained_model,
        Keras.layers.Flatten(),
        Keras.layers.Dense(1024, activation="relu", input_shape=(32, 32, 3)),
        Keras.layers.Dropout(0.3),
        Keras.layers.Dense(512, activation="relu"),
        Keras.layers.Dropout(0.3),
        Keras.layers.Dense(num_classes, activation="softmax"),
    ])

    optimizer = Keras.optimizers.Adam()
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["acc"])

    cbs = []

    sb = Keras.callbacks.ModelCheckpoint(model_name,
                                         monitor='val_accuracy',
                                         save_best_only=True,
                                         mode="max")
    cbs.append(sb)
    reduceLR = Keras.callbacks.ReduceLROnPlateau(monitor="val_acc",
                                                 factor=.01,
                                                 patience=3,
                                                 min_lr=1e-5)
    cbs.append(reduceLR)
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        validation_data=(x_test, y_test),
                        callbacks=cbs)
