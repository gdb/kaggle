#!/usr/bin/env python
import argparse
import cPickle
import glob
import gzip
import logging
import numpy
import PIL.Image
import sys

numpy.random.seed(51244)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

import clean

LOGGER_NAME = 'backend.%s'
logger = logging.getLogger(LOGGER_NAME % __name__)
logger.addHandler(logging.StreamHandler(sys.stderr))

def load_training(limit=None, neighbors=2):
    xs = []
    ys = []

    for path in glob.glob('../../train/*.png')[:limit]:
        patches, _ = clean.x_from_image(path, neighbors)
        solutions = clean.y_from_image(path, neighbors)

        xs.extend(patches)
        ys.extend(solutions)

    return xs, ys

def build_model(input_size):
    model = Sequential()
    model.add(Dense(input_size, 512, init='lecun_uniform'))
    model.add(Activation('tanh'))
    model.add(Dense(512, 256, init='lecun_uniform'))
    model.add(Activation('tanh'))
    model.add(Dense(256, 128, init='lecun_uniform'))
    model.add(Activation('tanh'))
    model.add(Dense(128, 64, init='lecun_uniform'))
    model.add(Activation('tanh'))
    model.add(Dense(64, 1, init='lecun_uniform'))
    model.add(Activation('tanh'))

    sgd = SGD(lr=0.01, momentum=0.9) # , nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

def split_training(xs, ys):
    joined = zip(xs, ys)
    numpy.random.shuffle(joined)
    xs = [x for x, _ in joined]
    ys = [y for _, y in joined]

    train_count = int(len(xs)*8/10.0)
    valid_count = int(len(xs)/10.0)
    test_count = valid_count

    res = (xs[:train_count], ys[:train_count], \
        xs[train_count:train_count+valid_count], ys[train_count:train_count+valid_count], \
        ys[train_count+valid_count:train_count+valid_count+test_count], ys[train_count+valid_count:train_count+valid_count+test_count])

    return [numpy.array(r) for r in res]

def train(limit, neighbors, epochs, batch_size):
    xs, ys = load_training(limit, neighbors)
    train_x, train_y, valid_x, valid_y, test_x, test_y = split_training(xs, ys)
    model = build_model(len(train_x[0]))
    model.fit(train_x, train_y,
              nb_epoch=epochs,
              batch_size=batch_size,
              show_accuracy=False,
              validation_data=(valid_x, valid_y))
    return model

def save_model(model, path):
    with open(path, 'w') as f:
        cPickle.dump(model, f)

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', help='Set verbosity.')
    parser.add_argument('-l', '--limit', help='Number of training images to load.', type=int)
    parser.add_argument('-n', '--neighbors', help='Number of neighbors to use for network.', type=int, default=2)
    parser.add_argument('-e', '--epochs', default=20, type=int, help='Number of epochs to run for.')
    parser.add_argument('-b', '--batch-size', default=10, type=int, help='The size of each minibatch.')
    parser.add_argument('path', help='Where to save the model.')
    args = parser.parse_args()

    if args.verbosity == 1:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 2:
        logger.setLevel(logging.DEBUG)

    model = train(limit=args.limit, neighbors=args.neighbors,
                  epochs=args.epochs, batch_size=args.batch_size)
    out = (model, {'neighbors': args.neighbors})
    save_model(out, args.path)

    return 0

if __name__ == '__main__':
    sys.exit(main())
