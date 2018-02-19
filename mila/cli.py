"""Command-line interface for performing preparation and training steps.
"""
import argparse
import logging

from . import config

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
    train(args.imagesize, args.epochs, args.batchsize, args.outputdir)


def predict_run(args):
    from .predict import predict
    predict(args.imagefile, args.modeldir)


def create_parser():
    """Creates a new argument parser for."""
    # pylint: disable=line-too-long
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Create first-level subcommand parsers
    prepare = subparsers.add_parser('prepare', help='prepare')
    train = subparsers.add_parser('train', help='train')
    predict = subparsers.add_parser('predict', help='predict')

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
    simple.add_argument('--outputdir', default='{}/simple'.format(config.OUTPUT_DIRECTORY), help='The name of the output directory for model output')
    simple.set_defaults(func=train_simple_run)

    parser.add_argument_group()
    predict.add_argument('imagefile', help='The location of a file to predict')
    predict.add_argument('modeldir', help='The directory where a trained model (h5) is located. It is assumed that the model is named model.h5')
    predict.set_defaults(func=predict_run)

    return parser


def main(args=None):
    """The main function which is run when the script is executed from the command-line"""
    logging.basicConfig(level=logging.DEBUG)
    parser = create_parser()
    args = parser.parse_args(args)
    args.func(args)


if __name__ == '__main__':
    main()
