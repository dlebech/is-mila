
# Is it Mila?

[![Build Status](https://travis-ci.org/dlebech/is-mila.svg?branch=master)](https://travis-ci.org/dlebech/is-mila)
[![codecov](https://codecov.io/gh/dlebech/is-mila/branch/master/graph/badge.svg)](https://codecov.io/gh/dlebech/is-mila)

A small project to answer the question: does this photo have Mila in it?

But it can also be used for more general image classification -- just don't expect too much :-)

![Mila](https://farm9.staticflickr.com/8275/28338981466_9bd1fbe82e_n.jpg)

## Install

Perhaps create a virtual environment, then:

```shell
pip install -r requirements.txt
```

### Secrets (optional)

Rename `env_sample.sh` to `env.sh`, put in the correct
credentials and run it:

```shell
source env.sh
```

## The command-line interface (CLI)

In general, most tasks for `is-mila` can be performed with the CLI tool. To explore the options:

```shell
python -m mila.cli -h
```

Some of the commands will be mentioned explicitly in the following sections.

## Prepare photo/image data

You can either use your own photos/images, or download some using the Flickr
API. See the options below.

### Bring your own photos

If you have a bunch of images you want to classify (e.g. the MNIST images),
create the directory `images/all` and put the images in sub-directories that
correspond to a single image category. For example, if you have cats and dogs
photos, organize them like this:

```
└── images
    └── all
        ├── cat
        |   ├── cat1.jpg
        |   └── cat2.jpg
        └── dog
            ├── dog1.jpg
            └── dog2.jpg
```

### Download from Flickr

The following command downloads photos from Flickr for the given user and
splits the photos in two groups based on the given tag:

```shell
python -m mila.cli prepare flickr --user hej --tags mila
```

After running the above command, two directories will be created, one
containing photos with the tag "mila" and one containing photos that are "not
mila" or "everything else" essentially:

```
└─ images
   └── all
       ├── mila
       └── not_mila
```

Multiple tags separates by comma will result in multiple category directories:

```shell
python -m mila.cli prepare flickr --user hej --tags cat,dog
```

```
└── images
    └── all
        ├── cat
        └── dog
```

### Prepare train and evaluation photo sets

After running the image preparation commands, the following command will
create a random train/validation split:

```shell
python -m mila.cli prepare traindata
```

Use the `--equalsplits` flag if you want each category to have the same number of samples.

## Train the network

The `train` subcommand is used for training on the prepared image data.

For example, to train a very simple network, simply run:

```shell
python -m mila.cli train simple
```

Using the default CLI parameters, this will create a trained model at `./output/simple/model.h5`.
The model will be quite bad at predicting stuff, but the command should be very fast to run :-)

For a slightly nicer model, perhaps increase the image size and the number of
epochs that it runs for:

```shell
python -m ismila.cli train simple --epochs 100 --imagesize 256,256
```

## Make predictions

After training a network, make predictions on images like this:

```shell
python -m mila.cli predict images/myimage.jpg output/simple
```

This will print the classification of `myimage.jpg` for the model stored in
the directory `output/simple`.

## Deploy the model

Besides making predictions from the command-line, is-mila contains a small
API server that can host the prediction, as well as a simple test page for
trying out new photos.

Start the server:

```shell
python -m mila.serve.app
```

This will start a webserver on port 8000. If you have a model called simple,
you can see it at `http://localhost:8000/model/simple`

### Using Docker

Docker is cool. The included Dockerfile will prepare a Docker image with all
currently trained models, if they are located in the default model output
location (`./output/*`).

For example, to deploy the models to a Heroku app (after following their [general instructions](https://devcenter.heroku.com/articles/container-registry-and-runtime)):

```shell
heroku container:push web
```

That's it! The web-server includes CORS headers by default, so you can access
the API from anywhere.

## Development

### Testing

```shell
python -m pytest --cov=mila tests/
```

### Explore models

Using [quiver](https://github.com/keplr-io/quiver), you can explore the
layers of the trained model in your browser:

```shell
python -m mila.cli explore ./path/to/image_dir ./path/to/model_dir
```
