"""Training module."""
import logging
import os

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from .. import config, util
from . import callback

logger = logging.getLogger(__name__)


def create_image_iterators(image_size: tuple, batch_size: int, categories: list, save_output=False) -> tuple:
    """Creates a pair of iterators for train and validation images."""
    # XXX: figure out why all examples use a 1/255 scaling. I mean, sure, it
    # scales the pixel to be between 0 and 1 but is that all?
    logger.info('Using {} classes'.format(len(categories)))
    if save_output:
        output_images = os.path.join(config.IMAGE_DIRECTORY, 'output')
        os.makedirs(output_images, exist_ok=True)
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
        batch_size=batch_size,
        save_to_dir=output_images if save_output else None,
        save_prefix='train'
    )

    # For validation, we don't want to shear, zoom or flip...
    validation_generator = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_generator.flow_from_directory(
        os.path.join(config.IMAGE_DIRECTORY, 'validation'),
        classes=categories,
        class_mode='binary' if len(categories) == 2 else 'categorical',
        target_size=image_size,
        batch_size=batch_size,
        save_to_dir=output_images if save_output else None,
        save_prefix='val'
    )

    return train_generator, validation_generator


def prepare_callbacks(model_filename, model_meta_filename, categories, batch_size,
                      tensorboard_logs=True,
                      save_model_checkpoints=True,
                      early_stopping=True):
    """Prepare training callbacks."""
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
        callbacks.append(ModelCheckpoint(model_filename, save_best_only=True, verbose=1))

    if early_stopping:
        callbacks.append(EarlyStopping(patience=20, verbose=1))

    callbacks.append(callback.ModelMetadata(model_meta_filename, categories))

    return callbacks