"""Command-line interface for performing preparation and training steps.
"""
import argparse
import logging

def flickr_run(args):
    from .prepare.flickr import run
    run(args.user, args.tags, args.limit)


def train_split_run(args):
    from .prepare.train_split import run
    run()


def create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Create first-level subcommand parsers
    prepare = subparsers.add_parser('prepare', help='prepare')
    train = subparsers.add_parser('train', help='train')

    # Datasources for data preparation.
    prepare_subparser = prepare.add_subparsers()
    flickr = prepare_subparser.add_parser('flickr', help='Fetch photos from Flickr based on a user and tags')
    flickr.add_argument('--user', help='The user to download photos for', required=True)
    flickr.add_argument('--tags', help='The tags to use to categorize the photos by', required=True)
    flickr.add_argument('--limit', help='The maximum number of photos to fetch', type=int, default=10)
    flickr.set_defaults(func=flickr_run)
    
    train_data = prepare_subparser.add_parser('traindata', help='Split the data into training and evaluation sets')
    train_data.set_defaults(func=train_split_run)

    return parser


def main(args=None):
    logging.basicConfig(level=logging.DEBUG)
    parser = create_parser()
    args = parser.parse_args(args)
    args.func(args)


if __name__ == '__main__':
    main()