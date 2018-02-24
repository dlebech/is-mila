"""Train a simple convolutional neural network for image classification."""
import json
import logging
import os
import typing

import keras.utils
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from .. import config, util
from . import create_image_iterators, prepare_callbacks


logger = logging.getLogger(__name__)


def setup_network(image_shape: tuple, num_classes: int):
    """Sets up the network architecture of CNN.
    """
    rows, cols = image_shape
    inputs = Input(shape=(rows, cols, 3))

    # 3 Layers of convolutions with max-pooling to reduce complexity.
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten and use a fully connected network.
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(
        1 if num_classes == 2 else num_classes,
        activation='sigmoid' if num_classes == 2 else 'softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train(image_size, epochs, batch_size, output_dir, tensorboard_logs='./logs', save_model_checkpoints=True, early_stopping=True):
    os.makedirs(output_dir, exist_ok=True)
    model_filename = os.path.join(output_dir, 'model.h5')
    model_meta_filename = os.path.join(output_dir, 'model_metadata.json')

    categories = util.find_categories('train')

    # Prepare the network model
    model = setup_network(image_size, len(categories))
    train_images, validation_images = create_image_iterators(image_size, batch_size, categories)

    callbacks = prepare_callbacks(
        model_filename,
        model_meta_filename,
        categories,
        batch_size,
        tensorboard_logs=tensorboard_logs,
        save_model_checkpoints=save_model_checkpoints,
        early_stopping=early_stopping)

    steps = util.num_samples('train')
    steps_per_epoch = steps // batch_size
    validation_steps = steps // batch_size // 3
    if steps_per_epoch < 1:
        steps_per_epoch = 1
    if validation_steps < 1:
        validation_steps = 1

    logger.info('Training network for {} epochs with normal step size {} and validation step size {}'
                .format(epochs, steps_per_epoch, validation_steps))

    # Train it!
    model.fit_generator(train_images,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_images,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Save some model data
    os.makedirs(output_dir, exist_ok=True)
    keras.utils.print_summary(model)
    if not save_model_checkpoints:
        model.save(model_filename)
