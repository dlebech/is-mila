"""Command-line interface for performing preparation and training steps.
"""
import argparse
import logging
import os

import tensorflow as tf
from keras.backend import set_session

from . import config

# This prevents cuDNN errors when training on GPU for some reason...
tensorflow_config = tf.ConfigProto()
tensorflow_config.gpu_options.allow_growth = True
sess = tf.Session(config=tensorflow_config)
set_session(sess)


def flickr_run(args):
    """Run Flickr preparation."""
    from .prepare.flickr import run
    run(args.user, args.tags, args.limit)


def train_split_run(args):
    # pylint: disable=unused-argument
    """Run train/validation split preparation."""
    from .prepare.train_split import run
    run(equal_splits=args.equalsplits)


def train_simple_run(args):
    """Run simple CNN training"""
    from .train.simple import train
    train(
        args.imagesize,
        args.epochs,
        args.batchsize,
        args.outputdir,
        use_class_weights=args.classweights,
        debug=args.debug,
        use_image_variations=args.imagevariations)


def train_mobilenet_run(args):
    """Run mobilenet CNN training"""
    from .train.mobilenet import train
    train(
        args.epochs,
        args.batchsize,
        args.outputdir,
        use_class_weights=args.classweights,
        use_image_variations=args.imagevariations)


def predict_run(args):
    from .predict import predict
    predict(args.imagefile, args.modeldir)


def explore_run(args):
    import tempfile
    from quiver_engine import server
    from . import predict
    model, meta = predict.load(args.modeldir)
    server.launch(model,
                  classes=meta['classes'],
                  input_folder=args.imagedir,
                  temp_folder=tempfile.mkdtemp(prefix='mila_quiver_'),
                  std=[255, 255, 255]) # This is a bit of a hack to make quiver scale the images


def evaluate_run(args):
    from .evaluate import evaluate
    evaluate(args.modeldir, args.imagedir)


def create_parser():
    """Creates a new argument parser for."""
    # pylint: disable=line-too-long
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Create first-level subcommand parsers
    prepare = subparsers.add_parser('prepare', help='prepare')
    train = subparsers.add_parser('train', help='train')
    predict = subparsers.add_parser('predict', help='predict')
    evaluate = subparsers.add_parser('evaluate', help='evaluate')
    explore = subparsers.add_parser('explore', help='explore')

    # Datasources for data preparation.
    prepare_subparser = prepare.add_subparsers()
    flickr = prepare_subparser.add_parser('flickr', help='Fetch photos from Flickr based on a user and tags')
    flickr.add_argument('--user', help='The user to download photos for', required=True)
    flickr.add_argument('--tags', help='The tags to use to categorize the photos by', required=True)
    flickr.add_argument('--limit', help='The maximum number of photos to fetch', type=int, default=10)
    flickr.set_defaults(func=flickr_run)

    train_data = prepare_subparser.add_parser('traindata', help='Split the data into training and evaluation sets')
    train_data.add_argument('--equalsplits', action='store_true', help='Split training categories into equal number of samples')
    train_data.set_defaults(func=train_split_run)

    def image_size_tuple(s):
        """Imagesize parser"""
        return tuple(int(i) for i in s.split(','))
    train_subparser = train.add_subparsers()
    simple = train_subparser.add_parser('simple', help='Train from scratch on a a very simple convolutional neural network. When using the defaults, training will usually be quite fast')
    simple.add_argument('--imagesize', type=image_size_tuple, default=(32, 32), help='The size that input images should be resized to. Has a big influence on training time')
    simple.add_argument('--epochs', type=int, default=10, help='Number of epochs to run the network for')
    simple.add_argument('--batchsize', type=int, default=32, help='The batch size for input images')
    simple.add_argument('--outputdir', default=os.path.join(config.OUTPUT_DIRECTORY, 'simple'), help='The name of the output directory for model output')
    simple.add_argument('--classweights', action='store_true', help='Use balanced class weigths')
    simple.add_argument('--debug', action='store_true', help='Use debug settings')
    simple.add_argument('--imagevariations', action='store_true', help='Create small image variations during training')
    simple.set_defaults(func=train_simple_run)

    mobilenet = train_subparser.add_parser('mobilenet', help='Train on top of MobileNet.')
    mobilenet.add_argument('--epochs', type=int, default=10, help='Number of epochs to run the network for')
    mobilenet.add_argument('--batchsize', type=int, default=32, help='The batch size for input images')
    mobilenet.add_argument('--outputdir', default=os.path.join(config.OUTPUT_DIRECTORY, 'mobilenet'), help='The name of the output directory for model output')
    mobilenet.add_argument('--classweights', action='store_true', help='Use balanced class weigths')
    mobilenet.add_argument('--imagevariations', action='store_true', help='Create small image variations during training')
    simple.set_defaults(func=train_simple_run)
    mobilenet.set_defaults(func=train_mobilenet_run)

    predict.add_argument('imagefile', help='The location of a file to predict')
    predict.add_argument('modeldir', help='The directory where a trained model (h5) is located. It is assumed that the model is named model.h5')
    predict.set_defaults(func=predict_run)

    explore.add_argument('imagedir', help='The location of image files to explore')
    explore.add_argument('modeldir', help='The directory where a trained model (h5) is located. It is assumed that the model is named model.h5')
    explore.set_defaults(func=explore_run)

    evaluate.add_argument('modeldir', help='The directory where a trained model (h5) is located. It is assumed that the model is named model.h5')
    evaluate.add_argument(
        '--imagedir',
        default='all',
        help='The image sub-directory for the image files to evaluate performance for',
        choices=['all', 'train', 'validation'])
    evaluate.set_defaults(func=evaluate_run)

    return parser


def main(args=None):
    """The main function which is run when the script is executed from the command-line"""
    logging.basicConfig(level=logging.INFO)
    parser = create_parser()
    args = parser.parse_args(args)
    args.func(args)


if __name__ == '__main__':
    main()
