"""Training module."""
import logging
import os

from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .. import config, util
from . import callback

logger = logging.getLogger(__name__)


def create_image_iterators(
    image_size: tuple,
    batch_size: int,
    categories: list,
    save_output=False,
    preprocessing_function=None,
    use_image_variations=False,
) -> tuple:
    """Creates a pair of iterators for train and validation images."""
    logger.info("Using {} classes".format(len(categories)))
    if save_output:
        output_images = os.path.join(config.IMAGE_DIRECTORY, "output")
        os.makedirs(output_images, exist_ok=True)
    train_generator = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        shear_range=0.2 if use_image_variations else 0.0,
        zoom_range=0.2 if use_image_variations else 0.0,
        horizontal_flip=True if use_image_variations else False,
    )
    train_generator = train_generator.flow_from_directory(
        os.path.join(config.IMAGE_DIRECTORY, "train"),
        classes=categories,
        class_mode="categorical",
        target_size=image_size,
        batch_size=batch_size,
        save_to_dir=output_images if save_output else None,
        save_prefix="train",
    )

    # For validation, we don't want to shear, zoom or flip...
    validation_generator = ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )
    validation_generator = validation_generator.flow_from_directory(
        os.path.join(config.IMAGE_DIRECTORY, "validation"),
        classes=categories,
        class_mode="categorical",
        target_size=image_size,
        batch_size=batch_size,
        save_to_dir=output_images if save_output else None,
        save_prefix="val",
    )

    return train_generator, validation_generator


def prepare_callbacks(
    model_filename,
    model_meta_filename,
    categories,
    batch_size,
    trainer,
    tensorboard_logs=True,
    save_model_checkpoints=True,
    early_stopping=True,
    reduce_learning_on_plateau=True,
    monitor="val_loss",
):
    """Prepare training callbacks."""
    callbacks = []
    if tensorboard_logs:
        callbacks.append(
            TensorBoard(
                log_dir="./logs",
                histogram_freq=0,
                batch_size=batch_size,
                write_graph=True,
                write_images=True,
            )
        )
    if save_model_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                model_filename, monitor=monitor, save_best_only=True, verbose=1
            )
        )

    if early_stopping:
        callbacks.append(EarlyStopping(monitor=monitor, patience=20, verbose=1))

    if reduce_learning_on_plateau:
        callbacks.append(ReduceLROnPlateau(monitor=monitor, verbose=1))

    callbacks.append(callback.ModelMetadata(model_meta_filename, categories, trainer))

    return callbacks
