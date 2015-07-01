#!/usr/bin/env python
import argparse
import cPickle
import logging
import numpy
import os.path
import PIL.Image
import sys

LOGGER_NAME = 'backend.%s'
logger = logging.getLogger(LOGGER_NAME % __name__)
logger.addHandler(logging.StreamHandler(sys.stderr))

def enlarged(ary, neighbors):
    height, width = ary.shape
    enlarged = numpy.zeros((height + 2 * neighbors, width + 2 * neighbors))

    # Fill in the corners
    enlarged[:neighbors,:neighbors] = ary[0, 0]
    enlarged[:neighbors,-neighbors:] = ary[0, -1]
    enlarged[-neighbors:,:neighbors] = ary[-1, 0]
    enlarged[-neighbors:,-neighbors:] = ary[-1, -1]

    # Fill in the edges
    enlarged[:neighbors, neighbors:-neighbors] = ary[0, :] # top
    enlarged[neighbors:-neighbors, :neighbors] = ary[:, 0][numpy.newaxis].T # left
    enlarged[-neighbors:, neighbors:-neighbors] = ary[-1, :] # top
    enlarged[neighbors:-neighbors, -neighbors:] = ary[:, -1][numpy.newaxis].T # right

    # Fill in the chewy center
    enlarged[neighbors:-neighbors,neighbors:-neighbors] = ary

    return enlarged

def patchify(enlarged, neighbors):
    output = []
    height, width = enlarged.shape
    for i in xrange(neighbors, height-neighbors):
        for j in xrange(neighbors, width-neighbors):
            patch = enlarged[i-neighbors:i+neighbors+1, j-neighbors:j+neighbors+1]
            output.append(patch.flatten())
    return output

# def to_range(ary):
#     return ary* 0.8/256.0 +0.1

# def from_range(ary):
#     return (ary - 0.1) * 256.0/0.8

def to_range(ary):
    return ary* 1.8/256.0 - 0.9

def from_range(ary):
    return (ary + 0.9) * 256.0/1.8

def x_from_image(source, neighbors):
    img = PIL.Image.open(source)
    ary = to_range(numpy.array(img))

    enlarged_img = enlarged(ary, neighbors)
    patches = patchify(enlarged_img, neighbors)

    return numpy.array(patches), ary.shape

def cleaned_path(path):
    name = os.path.basename(path)
    return os.path.join(os.path.dirname(path), '../train_cleaned', name)

def y_from_image(source, neighbors):
    output = []

    img = PIL.Image.open(cleaned_path(source))
    ary = to_range(numpy.array(img))
    height, width = ary.shape

    for i in xrange(height):
        for j in xrange(width):
            output.append(ary[i, j].flatten())
    return output

def image_from_y(y, shape):
    orig = from_range(y.reshape(shape))
    img = PIL.Image.fromarray(orig)
    return img

def load_model(path):
    with open(path) as f:
        return cPickle.load(f)

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('model', help='Model to use.')
    parser.add_argument('input', help='Image to clean.')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', help='Set verbosity.')
    parser.add_argument('-o', '--output', help='Where to write the cleaned image.')
    args = parser.parse_args()

    if args.verbosity == 1:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 2:
        logger.setLevel(logging.DEBUG)

    model, params = load_model(args.model)

    xs, shape = x_from_image(args.input, params['neighbors'])
    ys = model.predict(xs)
    img = image_from_y(ys, shape)

    if args.output is not None:
        img.convert('L').save(args.output)
    else:
        img.show()

    return 0

if __name__ == '__main__':
    sys.exit(main())
