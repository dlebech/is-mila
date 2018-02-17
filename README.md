
# Is it Mila?

[![Build Status](https://travis-ci.org/dlebech/is-mila.svg?branch=master)](https://travis-ci.org/dlebech/is-mila)
[![codecov](https://codecov.io/gh/dlebech/is-mila/branch/master/graph/badge.svg)](https://codecov.io/gh/dlebech/is-mila)

A small project to answer the question: does this photo have Mila in it?

But it can also be used for other stuff... perhaps :-)

![Mila](https://farm9.staticflickr.com/8275/28338981466_9bd1fbe82e_n.jpg)

## Install

Create a virtual environment, then:

```shell
pip install -r requirements.txt
```

Also, make sure to rename `env_sample.sh` to `env.sh`, put in the correct
credentials and run it:

```shell
source env.sh
```

## Prepare photo data

### Download from Flickr

The following command downloads photos from Flickr for the given user and
splits the photos in two groups based on the given tag:

```shell
$ python -m mila.cli prepare flickr --user hej --tags mila
```

After running the above command, two directories will be created, one
containing photos with the tag "mila" and one containing photos that are "not
mila" or "everything else" essentially:

```
├── images
│   ├── all
│   │   ├── mila
│   │   └── not_mila
```

Multiple tags separates by comma will result in multiple category directories:

```shell
$ python -m mila.cli prepare flickr --user hej --tags cat,dog
```

```
├── images
│   ├── all
│   │   ├── cat
│   │   └── dog
```

The photos can be manually curated as well, following the directory structure
seen here.

### Prepare train and evaluation photo sets

After running the image preparation commands, the following command will
create a random train/validation split:

```shell
$ python -m mila.cli prepare traindata
```

## Train the network

This is to be determined...

## Development

### Testing

```shell
python -m pytest --cov=mila tests/
```