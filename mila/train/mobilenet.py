"""Re-train the V."""
import logging
import math
import os

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.optimizers import SGD

from .. import util, config
from . import create_image_iterators, prepare_callbacks


logger = logging.getLogger(__name__)


def setup_network(image_size: tuple, num_classes: int):
    """Sets up the network architecture of the CNN.
    """
    # Use MobileNet as the base model, add a fully-connected layer and a
    # prediction layer for the various classes.
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=image_size + (3,))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze all layers except the ones we just added, i.e. freeze the base model
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
                  metrics=['accuracy'])

    return base_model, model


def train(epochs, batch_size, output_dir, use_class_weights=False, use_image_variations=False):
    # Default size for mobilenet is 224 by 224, so we'll just stick to that.
    image_size = (224, 224)

    os.makedirs(output_dir, exist_ok=True)
    model_filename = os.path.join(output_dir, 'model.h5')
    model_meta_filename = os.path.join(output_dir, 'model_metadata.json')

    categories = util.find_categories('train')

    # Prepare the network model
    base_model, model = setup_network(image_size, len(categories))
    train_images, validation_images = create_image_iterators(
        image_size,
        batch_size,
        categories,
        preprocessing_function=util.image_preprocessing_fun(config.TRAINER_MOBILENET),
        use_image_variations=use_image_variations)

    model_checkpoint_monitor = 'val_loss'
    validation_samples = util.num_samples('validation')
    if validation_samples == 0:
        logger.info('No validation samples, changing checkpoint to monitor "loss"')
        model_checkpoint_monitor = 'loss'

    callbacks = prepare_callbacks(
        model_filename,
        model_meta_filename,
        categories,
        batch_size,
        config.TRAINER_MOBILENET,
        monitor=model_checkpoint_monitor)

    steps = util.num_samples('train')
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
        class_weights = util.compute_class_weight(train_images.class_indices, 'train')
        reverse_mapping = {v: k for k, v in train_images.class_indices.items()}
        nice_class_weights = {reverse_mapping[i]: v for i, v in class_weights.items()}
        logger.info('Using class weights: {}'.format(nice_class_weights))

    logger.info('Training network for {} epochs with normal step size {} and validation step size {}'
                .format(epochs, steps_per_epoch, validation_steps))

    # Train it for a few epochs with the base model layers frozen
    model.fit_generator(
        train_images,
        steps_per_epoch=steps_per_epoch,
        epochs=5,
        validation_data=validation_images,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # Unfreeze the last 3 base model layers and train again.
    # The last 3 base model layers should consist of a conv2d, batch
    # normalization and activation layer.
    for layer in base_model.layers[-3:]:
        layer.trainable = True

    logger.info('Some layers unfrozen, continuing training')

    # Inspired from the Keras tutorial
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='binary_crossentropy' if len(categories) == 2 else 'categorical_crossentropy',
        metrics=['accuracy'])

    # Train it for a few epochs with the base model layers frozen
    model.fit_generator(
        train_images,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_images,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights
    )

    logger.info('Training set basic evaluation')
    res = model.evaluate_generator(train_images, steps=steps_per_epoch, verbose=1)
    for metric in zip(model.metrics_names, res):
        logger.info('Train {}: {}'.format(metric[0], metric[1]))

    if validation_samples > 0:
        res = model.evaluate_generator(validation_images, steps=validation_steps)
        logger.info('Validation set basic evaluation')
        for metric in zip(model.metrics_names, res):
            logger.info('Validation {}: {}'.format(metric[0], metric[1]))
