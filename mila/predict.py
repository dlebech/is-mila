"""Prediction functionality for all models"""
import os
import logging

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


logger = logging.getLogger(__name__)


def predict(image_file, model_output_dir):
    """Make predictions for the given image(s) and pre-trained model."""
    logger.info('Loading model from {}'.format(model_output_dir))
    model = load_model(os.path.join(model_output_dir, 'model.h5'))
    _, rows, cols, channels = model.inputs[0].shape
    assert channels == 3
    logger.info('Detected input image size: {}'.format((rows, cols)))

    files_to_predict = []
    if os.path.isfile(image_file):
        logger.debug('Single file detected')
        files_to_predict.append(image_file)
    elif os.path.isdir(image_file):
        logger.debug('Directory detected')
        files_to_predict.extend(
            os.path.join(image_file, filename)
            for filename in os.listdir(image_file))

    logger.info('Create predictions')
    for filename in files_to_predict:
        img = load_img(filename, target_size=(rows, cols))
        logger.info('{}: {}'.format(
            filename,
            model.predict(np.array([img_to_array(img)]))))
