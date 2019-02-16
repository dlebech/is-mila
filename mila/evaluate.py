"""Evaluation functions for an already trained model."""
import logging
import os
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .predict import predict
from . import util, config


logger = logging.getLogger(__name__)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def evaluate(model_output_dir, image_dir='all'):
    # Assume that the contents of the chosen image directory follow the
    # structure used for training, i.e. categories as sub-directories
    categories = util.find_categories(image_dir)
    y_true = []
    y_pred = []
    y_pred_softmax = []

    category_mapping = {category: i for i, category in enumerate(categories)}

    # Categories are in the correct index order.
    for idx, category in enumerate(categories):
        path = os.path.join(config.IMAGE_DIRECTORY, image_dir, category)
        predictions = predict(path, model_output_dir, cache_model=True)
        for prediction in predictions:
            y_true.append(idx)

            # Find the maximum prediction
            sorted_prediction_pairs = sorted(prediction.items(), key=lambda item: item[1], reverse=True)
            max_prediction = sorted_prediction_pairs[0]
            predicted_label = category_mapping[max_prediction[0]]
            predicted_softmax = max_prediction[1]
            y_pred.append(predicted_label)
            y_pred_softmax.append(predicted_softmax)

    labels = list(range(len(categories)))
    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=categories)
    logger.info('### Evaluation output ###')
    logger.info('Classification report:')
    logger.info('\n{}'.format(rep))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(cm, categories, normalize=True)
    plot_confusion_matrix(cm, categories)
    plt.show()
