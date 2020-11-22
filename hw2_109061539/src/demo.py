import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from src.dataset import WAFER_dataset as Dataset


def display_image(data):
    """Given ndarray of data, display it by using PIL
    Args:
        data: target ndarray of data
    Return:
        None
    """
    
    if len(data.shape) == 3 and data.shape[0] == 3:
        bdry, dfct, nrml = data[0, :, :], data[1, :, :], data[2, :, :]
    elif len(data.shape) == 3 and data.shape[2] == 3:
        bdry, dfct, nrml = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    elif len(data.shape) == 4 and data.shape[1] == 3:
        bdry, dfct, nrml = data[0, 0, :, :], data[0, 1, :, :], data[0, 2, :, :]
    elif len(data.shape) == 4 and data.shape[3] == 3:
        bdry, dfct, nrml = data[0, :, :, 0], data[0, :, :, 1], data[0, :, :, 2]
    else:
        raise ValueError("image shape {} is invalid".format(data.shape))

    plt.subplot(1, 3, 1)
    plt.imshow(bdry)
    plt.subplot(1, 3, 2)
    plt.imshow(dfct)
    plt.subplot(1, 3, 3)
    plt.imshow(nrml)
    plt.show()

def display_image_with_gen_data(data, gen_data):
    """Given ndarray of data and generated data, display them by using PIL
    Args:
        data: target ndarray of data
        gen_data: target ndarray of generated data
    Return:
        None
    """
    N = len(gen_data)
    bdry, dfct, nrml = data[0, :, :], data[1, :, :], data[2, :, :]
    
    plt.subplot(1 + N, 3, 1)
    plt.imshow(bdry)
    plt.subplot(1 + N, 3, 2)
    plt.imshow(dfct)
    plt.subplot(1 + N, 3, 3)
    plt.imshow(nrml)

    for i in range(N):
        gen_bdry = gen_data[i][0, :, :]
        gen_dfct = gen_data[i][1, :, :]
        gen_nrml = gen_data[i][2, :, :]
        
        plt.subplot(1 + N, 3, (i + 1) * 3 + 1)
        plt.imshow(gen_bdry)
        plt.subplot(1 + N, 3, (i + 1) * 3 + 2)
        plt.imshow(gen_dfct)
        plt.subplot(1 + N, 3, (i + 1) * 3 + 3)
        plt.imshow(gen_nrml)

    plt.show()

def demo(path_to_data, path_to_generated_data=None, index_to_demo=None):
    """Demo an image in the wafer dataset, and also the generated samples if them exist
    Args:
        path_to_data: directory to data
        path_to_generated_data: directory to generated data (set None if not exists)
        index_to_demo (int): index of image in the dataset, if None then random
    Return:
        None
    """
    
    dataset = Dataset(path_to_data)

    if path_to_generated_data is not None:
        show_gen_data = True
        gen_dataset = Dataset(path_to_generated_data, generated=True)
    else:
        show_gen_data = False

    if index_to_demo is None:
        index_to_demo = np.random.randint(0, len(dataset))
    elif not isinstance(index_to_demo, int):
        raise ValueError("index should be integer")
    elif index_to_demo > len(dataset):
        raise ValueError("index should between 0 and {}".format(len(dataset)))

    print('index to demo: {}'.format(index_to_demo))
    
    
    # read original data
    data, _ = dataset[index_to_demo]
    
    if show_gen_data:
        # read generated data
        gen_data = []
        for i in range(5):
            gd, _ = gen_dataset[index_to_demo * 5 + i]
            gen_data.append(gd)
        # display original data and generated data
        display_image_with_gen_data(data, gen_data)
    else:
        # display original data
        display_image(data)


if __name__ == "__main__":
    demo(path_to_data='./wafer', path_to_generated_data='./output/', index_to_demo=5)
