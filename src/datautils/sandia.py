import os
import sys
import scipy.ndimage as ndimage
from scipy.misc import imresize
import numpy as np
import pandas as pd

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


def get_part_of_image(img, grid_width, part_number):
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

    return imresize(np.copy(img[y_axis_start:y_axis_end, x_axis_start:x_axis_end, 0]), (IMG_SIZE, IMG_SIZE))


class SandiaDataProvider:
    """
    For iterating over the Sandia dataset
    """
    def __init__(self, which_set, matrix_types=None, shuffle_order=True, rng=None, dataset_home=_SANDIA_DATASET_HOME):
        """
        :param which_set(str): If 'test', we load the test set, otherwise we load the rest of the data
        :param matrix_types(tuple(str)): There are 4 types of matrices in the dataset - ('1_layer', '2_layer', '3_layer', 'logic')
            Only matrices from this the specified subsets will be loaded. If None, all types will be loaded
        :param shuffle_order(bool): Should data be shuffled between epochs
        :param rng(np.rand.RandomState): An RNG
        """
        if matrix_types is None:
            self.matrix_types = AVAILABLE_MATRIX_TYPES
        else:
            for mt in matrix_types:
                if mt not in AVAILABLE_MATRIX_TYPES:
                    raise ValueError('Unrecognized matrix type {0}. Available types are {1}'.format(
                        mt, str(AVAILABLE_MATRIX_TYPES)))
            self.matrix_types = AVAILABLE_MATRIX_TYPES

        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

        self.data_info = pd.read_csv(os.path.join(dataset_home, _DATASET_INFO_FILENAME))
        self.data_info = self.data_info[self.data_info['type'].isin(self.matrix_types)]
        self.shuffle_order = shuffle_order
        self.dataset_home = dataset_home

        if which_set == 'test':
            self.data_info = self.data_info[self.data_info.test]
        else:
            self.data_info = self.data_info[~self.data_info.test]

        # Shuffle and reidex from zero
        self.data_info.reset_index(inplace=True)
        self._load_data()

    def _shuffle(self):
        """
        Shuffle the data info - to be used between epochs
        Note we do not reset the index of the data_info
        """
        self.data_info = self.data_info.sample(frac=1, random_state=self.rng)

    def _load_data(self):
        """
        Traverse the data_info dataframe, split the images in the dataset and load these into numpy arrays
        """
        self.inputs = np.empty((len(self.data_info), IMG_SIZE, IMG_SIZE, QUESTION_IMAGE_COUNT + ANSWER_IMAGE_COUNT))
        self.targets = np.array(self.data_info.answer)

        for i, row in self.data_info.iterrows():
            q_img_path = os.path.join(self.dataset_home, row['type'], row['problem'] + '.png')
            a_img_path = os.path.join(self.dataset_home, row['type'], row['problem'] + '_Answers.png')
            q_img = ndimage.imread(q_img_path)
            a_img = ndimage.imread(a_img_path)

            for qi in range(QUESTION_IMAGE_COUNT):
                self.inputs[i, :, :, qi] = get_part_of_image(q_img, Q_GRID_WIDTH, qi)
            for ai in range(ANSWER_IMAGE_COUNT):
                self.inputs[i, :, :, QUESTION_IMAGE_COUNT + ai] = get_part_of_image(a_img, A_GRID_WIDTH, ai)

            self.targets[i] = row['answer']

    def get_batch_iterator(self, batch_size):
        """
        Returns a generator object that yields batches (inputs, targets), iterating through the whole dataset
        :param batch_size: The number of samples in each batch
        """
        if self.shuffle_order:
            self._shuffle()

        num_batches = int(np.ceil(len(self.data_info) / batch_size))

        # Assign each example to a batch
        self.data_info.batch = np.arange(len(self.data_info)) % num_batches

        for bi in range(num_batches):
            batch_indices = self.data_info[self.data_info.batch == bi].index.values
            inputs = self.inputs[batch_indices]
            targets = self.targets[batch_indices]

            yield (inputs, targets)

    def get_bagging_batch_iterator(self, batch_size):
        """
        Returns a generator object that yields batches (inputs, targets), selecting a random subset of the samples for
        each batch
        :param batch_size: The number of samples in each batch
        """
        num_batches = int(np.ceil(len(self.data_info) / batch_size))

        for bi in range(num_batches):
            batch_indices = np.random.randint(0, len(self.data_info), batch_size)
            inputs = self.inputs[batch_indices]
            targets = self.targets[batch_indices]

            yield (inputs, targets)