import six.moves.cPickle as pickle
import gzip
import os

import numpy as np

import theano
import theano.tensor as T
from IPython import embed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

from untapped.utilities import load_url


def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def load_data(dataset='examples/datasets/mnist/mnist.pkl.gz',remove_mean=False,cat=True,binary=True,
              close_corners=True,fuzzy_corners=False,half_corners=False):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    train_set, valid_set, test_set = load_url('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz',dataset)

    # # Download the MNIST dataset if it is not present
    # data_dir, data_file = os.path.split(dataset)
    # if data_dir == "" and not os.path.isfile(dataset):
    #     # Check if dataset is in the data directory.
    #     new_path = os.path.join(
    #         os.path.split(__file__)[0],
    #         dataset
    #     )
    #     if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
    #         dataset = new_path

    # if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
    #     from six.moves import urllib
    #     origin = (
    #         'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    #     )
    #     print('Downloading data from %s' % origin)
    #     urllib.request.urlretrieve(origin, dataset)

    # print('... loading data')

    # # Load the dataset
    # with gzip.open(dataset, 'rb') as f:
    #     try:
    #         train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    #     except:
    #         train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    # test_set_x, test_set_y = shared_dataset(test_set)
    # valid_set_x, valid_set_y = shared_dataset(valid_set)
    # train_set_x, train_set_y = shared_dataset(train_set)

    # unsup_x contains all images
    # unsup_y contains corners of simplex for [5-9]
    # sup_x, sup_y contains labeled pairs for [0-4]
    # valid_x, valid_y contains labeled pairs for [5-9]

    test_x, test_y = test_set
    valid_x, valid_y = valid_set
    train_x, train_y = train_set

    x = np.vstack((valid_x,test_x,train_x))
    # x = 1-x  # black background (0's) with white digits (1's)
    if binary:
        x = np.round(x)

    # mean zero
    ux = x.mean(axis=0)
    if remove_mean:
        x -= ux

    _y = np.hstack((valid_y,test_y,train_y))
    num_y = len(np.unique(_y))
    N = len(_y)

    y = np.zeros((N,num_y))
    y[np.arange(N),_y] = 1
    if not cat:
        y = y[:,:-1]

    inds_lo = _y < 5
    inds_hi = _y >= 5

    inds_unsup_x = np.random.choice(x.shape[0],size=20000)
    x_unsup = x[inds_unsup_x]

    num_y_unsup = 20000
    mx = 0.95

    if half_corners:
        rest = (1-mx)/(num_y//2-1)
        if close_corners:
            I = np.ones((num_y//2,num_y//2))*rest + np.diag(np.ones(num_y//2))*(mx-rest)
        else:
            I = np.eye(num_y//2)
        y_unsup = np.hstack((np.zeros((num_y//2,num_y//2)),I))
        if fuzzy_corners:
            fuzzy = np.random.rand(num_y_unsup,num_y)*.01 + np.tile(y_unsup,(num_y_unsup//(num_y//2),1))
            y_unsup = fuzzy/np.sum(fuzzy,axis=1,keepdims=True)
    else:
        rest = (1-mx)/(num_y-1)
        if close_corners:
            I = np.ones((num_y,num_y))*rest + np.diag(np.ones(num_y))*(mx-rest)
        else:
            I = np.eye(num_y)
        y_unsup = I
        if fuzzy_corners:
            fuzzy = np.random.rand(num_y_unsup,num_y)*.01 + np.tile(y_unsup,(num_y_unsup//(num_y),1))
            y_unsup = fuzzy/np.sum(fuzzy,axis=1,keepdims=True)
    if not cat:
        y_unsup = y_unsup[:,:-1]

    x_train = x[inds_lo]
    y_train = y[inds_lo]

    x_valid = x[inds_hi]
    y_valid = y[inds_hi]

    xy = (x_train, y_train, x_valid, y_valid, x_unsup, y_unsup)
    xy_names = ('x_train', 'y_train', 'x_valid', 'y_valid', 'x_unsup', 'y_unsup')
    print('Data Shapes:')
    for name, d in zip(xy_names,xy):
        print(name,d.shape)

    names = [str(i) for i in range(num_y)]
    cmap = get_cmap(num_y)
    colors = [cmap(i) for i in range(num_y)]

    return xy, ux, names, colors




