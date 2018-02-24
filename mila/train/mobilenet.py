"""Re-train the V."""
import json
import logging
import os
import typing

import keras.utils
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np

from .. import util
from . import create_image_iterators, prepare_callbacks


logger = logging.getLogger(__name__)


def setup_network(image_size: tuple, num_classes: int):
    """Sets up the network architecture of the CNN.
    """
    # Use MobileNet as the base model, add a fully-connected layer and a
    # prediction layer for the various classes.
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=image_size + (3,))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(
        1 if num_classes == 2 else num_classes,
        activation='sigmoid' if num_classes == 2 else 'softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze all layers except the ones we just added, i.e. freeze the base model
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
                  metrics=['binary_accuracy'])

    return base_model, model


def train(epochs, batch_size, output_dir):
    # Default size for mobile nets is 224 by 224, so we'll just stick to that.
    image_size = (224, 224)

    os.makedirs(output_dir, exist_ok=True)
    model_filename = os.path.join(output_dir, 'model.h5')
    model_meta_filename = os.path.join(output_dir, 'model_metadata.json')

    categories = util.find_categories('train')

    # Prepare the network model
    base_model, model = setup_network(image_size, len(categories))
    train_images, validation_images = create_image_iterators(image_size, batch_size, categories)

    callbacks = prepare_callbacks(model_filename, model_meta_filename, categories, batch_size)

    steps = util.num_samples('train')
    steps_per_epoch = steps // batch_size
    validation_steps = steps // batch_size // 3
    if steps_per_epoch < 1:
        steps_per_epoch = 1
    if validation_steps < 1:
        validation_steps = 1

    logger.info('Training network for {} epochs with normal step size {} and validation step size {}'
                .format(epochs, steps_per_epoch, validation_steps))

    # Train it for a few epochs with the base model layers frozen
    model.fit_generator(train_images,
        steps_per_epoch=steps_per_epoch,
        epochs=5,
        validation_data=validation_images,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    # Unfreeze the last 3 base model layers and train again.
    # The last 3 base model layers should consist of a conv2d, batch
    # normalization and activation layer.
    for layer in base_model.layers[-3:]:
        layer.trainable = True

    # Inspired from the Keras tutorial
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='binary_crossentropy' if len(categories) == 2 else 'categorical_crossentropy',
        metrics=['accuracy'])

    # Train it for a few epochs with the base model layers frozen
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
