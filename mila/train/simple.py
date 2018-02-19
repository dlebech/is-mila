"""Train a simple convolutional neural network for image classification."""
import json
import logging
import os
import typing

import keras.utils
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from .. import config, util


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


def create_image_iterators(image_size: tuple, batch_size: int, categories: list) -> tuple:
    """Creates a pair of iterators for train and validation images."""
    # XXX: figure out why all examples use a 1/255 scaling. I mean, sure, it
    # scales the pixel to be between 0 and 1 but is that all?
    train_generator = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_generator.flow_from_directory(
        os.path.join(config.IMAGE_DIRECTORY, 'train'),
        classes=categories,
        class_mode='binary' if len(categories) == 2 else 'categorical',
        target_size=image_size,
        batch_size=batch_size
    )

    # For validation, we don't want to shear, zoom or flip...
    validation_generator = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_generator.flow_from_directory(
        os.path.join(config.IMAGE_DIRECTORY, 'validation'),
        classes=categories,
        class_mode='binary' if len(categories) == 2 else 'categorical',
        target_size=image_size,
        batch_size=batch_size
    )

    return train_generator, validation_generator


def train(image_size, epochs, batch_size, output_dir, tensorboard_logs='./logs', save_model_checkpoints=True):
    os.makedirs(output_dir, exist_ok=True)

    categories = util.find_categories('train')

    # Prepare the network model
    model = setup_network(image_size, len(categories))
    train_images, validation_images = create_image_iterators(image_size, batch_size, categories)

    callbacks = []
    if tensorboard_logs:
        callbacks.append(
            TensorBoard(
                log_dir='./logs',
                histogram_freq=0,
                batch_size=batch_size,
                write_graph=True,
                write_images=True
            )
        )
    if save_model_checkpoints:
        checkpoint_file = os.path.join(output_dir, 'model.{epoch:02d}-{val_loss:.2f}.h5')
        callbacks.append(ModelCheckpoint(checkpoint_file))

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
    # XXX: currently doesn't work with default graphviz and pydot packages.
    #keras.utils.plot_model(model, os.path.join(output_dir, 'model.png'))
    model.save(os.path.join(output_dir, 'model.h5'))

    # Save some extra model metadata so we will know later what a prediction
    # means.
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump({
            'classes': categories
        }, f)
