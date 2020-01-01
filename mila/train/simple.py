"""Train a simple convolutional neural network for image classification."""
import logging
import math
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from .. import config, util
from . import create_image_iterators, prepare_callbacks


logger = logging.getLogger(__name__)


def setup_network(image_shape: tuple, num_classes: int):
    """Sets up the network architecture of CNN.
    """
    rows, cols = image_shape
    inputs = Input(shape=(rows, cols, 3))

    # 3 Layers of convolutions with max-pooling to reduce complexity.
    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten and use a fully connected network.
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(
        optimizer="sgd",
        loss="binary_crossentropy" if num_classes == 2 else "categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(
    image_size,
    epochs,
    batch_size,
    output_dir,
    tensorboard_logs="./logs",
    save_model_checkpoints=True,
    early_stopping=True,
    use_class_weights=False,
    debug=False,
    use_image_variations=False,
):
    os.makedirs(output_dir, exist_ok=True)
    model_filename = os.path.join(output_dir, "model.h5")
    model_meta_filename = os.path.join(output_dir, "model_metadata.json")

    categories = util.find_categories("train")

    # Prepare the network model
    model = setup_network(image_size, len(categories))
    train_images, validation_images = create_image_iterators(
        image_size,
        batch_size,
        categories,
        save_output=debug,
        preprocessing_function=util.image_preprocessing_fun(config.TRAINER_SIMPLE),
        use_image_variations=use_image_variations,
    )

    # Sometimes there's no validation images. In those case, the model
    # checkpoint should monitor loss instead of validation loss
    model_checkpoint_monitor = "val_loss"
    validation_samples = util.num_samples("validation")
    if validation_samples == 0:
        logger.info('No validation samples, changing checkpoint to monitor "loss"')
        model_checkpoint_monitor = "loss"

    callbacks = prepare_callbacks(
        model_filename,
        model_meta_filename,
        categories,
        batch_size,
        config.TRAINER_SIMPLE,
        tensorboard_logs=tensorboard_logs,
        save_model_checkpoints=save_model_checkpoints,
        early_stopping=early_stopping,
        monitor=model_checkpoint_monitor,
    )

    steps = util.num_samples("train")
    validation_steps = validation_samples
    steps_per_epoch = math.ceil(steps / batch_size)
    validation_steps = math.ceil(validation_steps / batch_size)
    if steps_per_epoch < 1:
        steps_per_epoch = 1

    # Clear validation sample settings at this point, if needed
    if validation_samples == 0:
        validation_steps = None
        validation_images = None
    elif validation_steps < 1:
        validation_steps = 1

    class_weights = None
    if use_class_weights:
        class_weights = util.compute_class_weight(train_images.class_indices, "train")
        reverse_mapping = {v: k for k, v in train_images.class_indices.items()}
        nice_class_weights = {reverse_mapping[i]: v for i, v in class_weights.items()}
        logger.info("Using class weights: {}".format(nice_class_weights))

    logger.info(
        "Training network for {} epochs with normal step size {} and validation step size {}".format(
            epochs, steps_per_epoch, validation_steps
        )
    )

    # Train it!
    model.fit_generator(
        train_images,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_images,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights,
        use_multiprocessing=False,
        workers=1,
    )

    # Save some model data
    os.makedirs(output_dir, exist_ok=True)
    if not save_model_checkpoints:
        model.save(model_filename)

    res = model.evaluate_generator(train_images, steps=steps_per_epoch)
    logger.info("Training set basic evaluation")
    for metric in zip(model.metrics_names, res):
        logger.info("Train {}: {}".format(metric[0], metric[1]))

    if validation_samples > 0:
        res = model.evaluate_generator(validation_images, steps=validation_steps)
        logger.info("Validation set basic evaluation")
        for metric in zip(model.metrics_names, res):
            logger.info("Validation {}: {}".format(metric[0], metric[1]))
