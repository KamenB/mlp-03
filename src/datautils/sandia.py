import os
import sys
import scipy.ndimage as ndimage
from scipy.misc import imresize
import numpy as np
import pandas as pd
import torch

# TODO: Fix sibling directory imports
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cwd, '../..'))

from src.config import DEFAULT_SEED

_SANDIA_DATASET_HOME = 'datasets/sandia'
AVAILABLE_MATRIX_TYPES = ('1_layer', '2_layer', '3_layer', 'logic')
_DATASET_INFO_FILENAME = 'data_info.csv'

# Questions are on a 3x3 grid, answers are on a 2x4 grid
QUESTION_IMAGE_COUNT = 8
ANSWER_IMAGE_COUNT = 8
Q_GRID_HEIGHT = 3
Q_GRID_WIDTH = 3
A_GRID_HEIGHT = 2
A_GRID_WIDTH = 4
# The images in the grids are padded with a 2-pixel border on all sides and are 100x100 pixels each
BORDER_SIZE = 2
SUBIMAGE_SIZE = 100

# Our data provider will return images of size IMG_SIZExIMG_SIZE
IMG_SIZE = 64


def get_part_of_image(img, grid_width, part_number, result_img_size=IMG_SIZE):
    """
    Cut a part of a grid image - used to extract idividual pieces from grid images in the Sandia dataset
    :param img(np.array): The original grid image
    :param grid_width(int): how many items per row of the grid
    :param part_number(int, 0-based): which item to return (we count left to right, top to bottom, starting from 0)
    :return(np.array): The cropped subimage, resized to IMG_SIZExIMG_SIZE
    """
    row = part_number // grid_width
    col = part_number % grid_width

    x_axis_start = int((1 + col) * BORDER_SIZE + col * SUBIMAGE_SIZE)
    x_axis_end = int(x_axis_start + SUBIMAGE_SIZE)

    y_axis_start = int((1 + row) * BORDER_SIZE + row * SUBIMAGE_SIZE)
    y_axis_end = int(y_axis_start + SUBIMAGE_SIZE)

    return imresize(np.copy(img[y_axis_start:y_axis_end, x_axis_start:x_axis_end, 0]), (result_img_size, result_img_size))


class SandiaDataProvider:
    """
    For iterating over the Sandia dataset
    """
    def __init__(self, which_set, matrix_types=None, shuffle_order=True, rng=None, dataset_home=_SANDIA_DATASET_HOME,
                 img_size=IMG_SIZE, normalize_mean=None, normalize_sd=None):
        """
        :param which_set(str): If 'test', we load the test set, otherwise we load the rest of the data
        :param matrix_types(tuple(str)): There are 4 types of matrices in the dataset - ('1_layer', '2_layer', '3_layer', 'logic')
            Only matrices from this the specified subsets will be loaded. If None, all types will be loaded
        :param shuffle_order(bool): Should data be shuffled between epochs
        :param rng(np.rand.RandomState): An RNG
        """
        self.normalize_sd = normalize_sd
        self.normalize_mean = normalize_mean
        if matrix_types is None:
            self.matrix_types = AVAILABLE_MATRIX_TYPES
        else:
            for mt in matrix_types:
                if mt not in AVAILABLE_MATRIX_TYPES:
                    raise ValueError('Unrecognized matrix type {0}. Available types are {1}'.format(
                        mt, str(AVAILABLE_MATRIX_TYPES)))
            self.matrix_types = matrix_types

        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

        self.data_info = pd.read_csv(os.path.join(dataset_home, _DATASET_INFO_FILENAME))
        self.data_info = self.data_info[self.data_info['type'].isin(self.matrix_types)]
        self.shuffle_order = shuffle_order
        self.dataset_home = dataset_home
        self.img_size=img_size

        if which_set == 'test':
            self.data_info = self.data_info[self.data_info.test]
        elif which_set == 'valid':
            self.data_info = self.data_info[self.data_info.valid]
        elif which_set == 'train':
            self.data_info = self.data_info[(~self.data_info.valid) & (~self.data_info.test)]
        elif which_set is not None:
            raise ValueError(f'Invalid value for which_set: {which_set} - must be one of [test, valid, train, None]')


        # Shuffle and reidex from zero
        self.data_info.reset_index(inplace=True)
        self._load_data(img_size)

        self.flat_inputs = None

    def _shuffle(self):
        """
        Shuffle the data info - to be used between epochs
        Note we do not reset the index of the data_info
        """
        self.data_info = self.data_info.sample(frac=1, random_state=self.rng)

    def _load_data(self, img_size=IMG_SIZE):
        """
        Traverse the data_info dataframe, split the images in the dataset and load these into numpy arrays
        """
        self.inputs = np.empty((len(self.data_info), img_size, img_size, QUESTION_IMAGE_COUNT + ANSWER_IMAGE_COUNT))
        # Reindex so that answers are zero-based (take 1)
        self.targets = np.array(self.data_info.answer) - 1
        self.type_targets = np.array(self.data_info.type_id)

        for i, row in self.data_info.iterrows():
            q_img_path = os.path.join(self.dataset_home, row['type'], row['problem'] + '.png')
            a_img_path = os.path.join(self.dataset_home, row['type'], row['problem'] + '_Answers.png')
            q_img = ndimage.imread(q_img_path)
            a_img = ndimage.imread(a_img_path)

            for qi in range(QUESTION_IMAGE_COUNT):
                self.inputs[i, :, :, qi] = get_part_of_image(q_img, Q_GRID_WIDTH, qi, img_size)
            for ai in range(ANSWER_IMAGE_COUNT):
                self.inputs[i, :, :, QUESTION_IMAGE_COUNT + ai] = get_part_of_image(a_img, A_GRID_WIDTH, ai, img_size)

        if self.normalize_mean is None:
            self.normalize_mean = self.inputs.mean()
        if self.normalize_sd is None:
            self.normalize_sd = np.sqrt(self.inputs.var())

        # Normalize inputs - zero mean, unit variance
        self.inputs = (self.inputs - self.normalize_mean) / self.normalize_sd

    def size(self):
        return len(self.data_info)

    def get_batch_iterator(self, batch_size, type_targets=False, transpose_inputs=False, separate_inputs=False):
        """
        Returns a generator object that yields batches (inputs, targets), iterating through the whole dataset
        :param batch_size: The number of samples in each batch
        """
        if self.shuffle_order:
            self._shuffle()

        # Truncate data - if we have 84 datapoints and a batch_size of 10, we will return 8 batches of size 10
        data_info_copy = self.data_info.copy(deep=True)
        if len(self.data_info) % batch_size > 0:
            data_info_copy = data_info_copy.iloc[:-(len(self.data_info) % batch_size)]
        num_batches = int(np.floor(len(self.data_info) / batch_size))

        # Assign each example to a batch
        data_info_copy.loc[:, 'batch'] = np.arange(len(data_info_copy)) % num_batches

        for bi in range(num_batches):
            batch_indices = data_info_copy[data_info_copy.batch == bi].index.values
            if type_targets:
                # Only return question images
                inputs = self.inputs[batch_indices][:, :, :, :QUESTION_IMAGE_COUNT]
                targets = self.type_targets[batch_indices]
            else:
                inputs = self.inputs[batch_indices]
                targets = self.targets[batch_indices]

            if transpose_inputs:
                inputs = inputs.transpose(0, 3, 1, 2)
                if separate_inputs:
                    inputs = inputs[:, :QUESTION_IMAGE_COUNT, ...], inputs[:, QUESTION_IMAGE_COUNT:, ...]
            else:
                if separate_inputs:
                    inputs = inputs[..., :QUESTION_IMAGE_COUNT], inputs[..., QUESTION_IMAGE_COUNT:]
            yield inputs, targets

    def get_image_batch_iterator(self, batch_size, img_as_vector=False):
        if self.flat_inputs is None:
            self.flat_inputs = self.inputs.transpose(3,0,1,2).reshape(-1, 1, self.img_size, self.img_size)

        if self.shuffle_order:
            np.random.shuffle(self.flat_inputs)

        num_batches = int(np.ceil(len(self.flat_inputs) / batch_size))

        for bi in range(num_batches):
            if img_as_vector:
                yield torch.from_numpy(self.flat_inputs[batch_size * bi: batch_size * (bi + 1)].reshape(
                    -1, self.img_size ** 2).astype('float32')), None
            else:
                yield torch.from_numpy(self.flat_inputs[batch_size * bi: batch_size * (bi + 1)].astype('float32')), None

    def get_bagging_batch_iterator(self, batch_size):
        """
        Returns a generator object that yields batches (inputs, targets), selecting a random subset of the samples for
        each batch
        :param batch_size: The number of samples in each batch
        """
        num_batches = int(np.ceil(self.size() / batch_size))

        for bi in range(num_batches):
            batch_indices = np.random.randint(0, self.size(), batch_size)
            inputs = self.inputs[batch_indices]
            targets = self.targets[batch_indices]

            yield (inputs, targets)