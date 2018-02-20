"""Prediction functionality for all models"""
import io
import json
import os
import logging

import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


logger = logging.getLogger(__name__)


def load(model_output_dir):
    logger.info('Loading model from {}'.format(model_output_dir))
    model_filename = os.path.join(model_output_dir, 'model.h5')
    metadata_filename = os.path.join(model_output_dir, 'model_metadata.json')
    if not os.path.exists(model_filename):
        return None, None

    model = load_model(model_filename)

    meta = None
    with open(metadata_filename) as f:
        meta = json.load(f)

    return model, meta


def model_info(model_output_dir):
    model, meta = load(model_output_dir)
    if not model:
        return None

    batch_size, rows, cols, channels = model.inputs[0].get_shape().as_list()
    return {
        'network': {
            'batch_size': batch_size,
            'image': {
                'rows': rows,
                'cols': cols,
                'color_channels': channels
            }
        },
        'meta': meta
    }


def prepare_image(image_file, target_size):
    if isinstance(image_file, bytes):
        logger.debug('Raw byte image detected')
        img = load_img(io.BytesIO(image_file), target_size=target_size)
        yield ('N/A',img_to_array(img))
    elif os.path.isfile(image_file):
        logger.debug('Single file detected')
        img = load_img(image_file, target_size=target_size)
        yield (image_file, img_to_array(img))
    elif os.path.isdir(image_file):
        logger.debug('Directory detected')
        for filename in sorted(os.listdir(image_file)):
            filename = os.path.join(image_file, filename)
            img = load_img(filename, target_size=target_size)
            yield (filename, img_to_array(img))

model_cache = {}

def predict(image_file, model_output_dir, cache_model=False):
    """Make predictions for the given image(s) and pre-trained model.

    :param image_file: An image filename, directory containing multiple
    images or an already loaded byte-like image.
    :param model_output_dir: A directory where the model exists
    :param cache_model: Determines whether or not to store the loaded model
    in memory. This can help speed up future predictions

    """
    if model_output_dir in model_cache:
        logger.info('Using cached model')
        model, meta = model_cache.get(model_output_dir)
    else:
        logger.info('Loading model from model file')
        model, meta = load(model_output_dir)

    if cache_model:
        logger.debug('Saving model in model cache')
        model_cache[model_output_dir] = (model, meta)

    classes = meta['classes']
    _, rows, cols, channels = model.inputs[0].get_shape().as_list() # Tensor has as_list
    assert channels == 3
    assert len(classes) >= 2
    logger.info('Detected input image size: {}'.format((rows, cols)))
    logger.debug('Classnames: {}'.format(classes))

    image_iterator = prepare_image(image_file, (rows, cols))

    logger.info('Create predictions')
    predictions = []

    for i, (img_file, img) in enumerate(image_iterator):
        prediction = model.predict(np.array([img]))
        prediction = prediction[0].tolist() # Numpy array has tolist
        outcomes = None
        # In the case where there is only one prediction, a 0 means the first
        # class is true and a 1 means the second class is true.
        # In other cases, there will be a prediction for each label
        if len(prediction) == 1:
            outcome_index = round(prediction[0])
            not_outcome_index = (outcome_index + 1) % 2
            outcomes = {
                classes[outcome_index]: 1,
                classes[not_outcome_index]: 0
            }
        else:
            outcomes = { classes[i]: prediction[i] for i, v in enumerate(prediction) }

        logger.info('{}, {}: {}'.format(i, img_file, outcomes))
        predictions.append(outcomes)

    return predictions
