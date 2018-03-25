from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable


def show_batch_of_images(img_batch, fig):
    batch_size, im_height, im_width = img_batch.shape
    # calculate no. columns per grid row to give square grid
    grid_size = int(np.ceil(batch_size ** 0.5))
    # intialise empty array to tile image grid into
    tiled = np.zeros((im_height * grid_size,
                      im_width * grid_size))
    # iterate over images in batch + indexes within batch
    for i, img in enumerate(img_batch):
        # calculate grid row and column indices
        c, r = i % grid_size, i // grid_size
        tiled[r * im_height:(r + 1) * im_height,
        c * im_height:(c + 1) * im_height] = img
    ax = fig.add_subplot(111)
    ax.imshow(tiled, cmap='Greys')
    ax.axis('off')
    fig.tight_layout()
    plt.show()
    return fig, ax


def show_grid_of_images(img_batch, img_size=(1,1), grid_size=None, show=True):
    # How many squares for a square grid that can fit all images
    if grid_size == None:
        grid_size = math.ceil(math.sqrt(len(img_batch)))
        grid_size = (grid_size, grid_size)
    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(img_size[0] * grid_size[0], img_size[1] * grid_size[1]))
    # Turn the 2d array of axes to a 1d array
    axs = axs.flatten()
    for i, img in enumerate(img_batch):
        axs[i].imshow(img.reshape(28,28))
    # Do this separately in case the number of images we want to show is not a perfect square
    for i in range(grid_size[0] * grid_size[1]):
        axs[i].axis('off')
    plt.show()


def show_matrix(inputs, targets, decoded_inputs, decoded_predictions):
    '''
        Input:
            inputs                  - batch_sizex8xim_widthxim_height
            targets                 - batch_sizex1xim_widthxim_height
            decoded_inputs          - batch_sizex8xim_widthxim_height
            decoded_predictions     - batch_sizex1xim_widthxim_height
    '''
    for batch_idx in range(inputs.size(0)):
        inputs_np = inputs[batch_idx].cpu().data.numpy()
        decoded_inputs_np = decoded_inputs[batch_idx].cpu().data.numpy()
        targets_np = targets[batch_idx:batch_idx+1].cpu().data.numpy()
        
        decoded_predictions_np = decoded_predictions[batch_idx:batch_idx+1].cpu().data.numpy()
        
        inputs_np = np.concatenate([inputs_np, targets_np])
        decoded_np = np.concatenate([decoded_inputs_np, decoded_predictions_np])
        
        show_grid_of_images(np.concatenate([inputs_np, decoded_np]), img_size=(9, 0.5), grid_size=(2, 9), show=False)
    plt.show()


def make_vars(np_arrays, dtype_names, use_cuda):
    dtype_dict = {
        'float': torch.cuda.FloatTensor if use_cuda else torch.FloatTensor,
        'long': torch.cuda.LongTensor if use_cuda else torch.LongTensor,
    }
    tensors = [torch.from_numpy(np_array).type(dtype_dict[dtype_name]) for np_array, dtype_name in zip(np_arrays, dtype_names)]
    return [Variable(x) for x in tensors]


def dict_of_lists_to_list_of_dicts(dict_of_lists):
    combinations = product(*(dict_of_lists.values()))
    return [dict(zip(dict_of_lists.keys(), combo)) for combo in combinations]