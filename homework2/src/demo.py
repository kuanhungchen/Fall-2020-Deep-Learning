import os
import numpy as np
from matplotlib import pyplot as plt

from src.dataset import WAFER_dataset as Dataset


LABEL2TEXT = {
    0: 'Central',
    1: 'Donut',
    2: 'Edge-Loc',
    3: 'Edge-Ring',
    4: 'Loc',
    5: 'Near-full',
    6: 'Random',
    7: 'Scratch',
    8: 'None'
}

def display_image_with_gen_data(data, gen_data, label=None):
    """Display single image and generated samples given their ndarray by using
    matplotlib package.

    Args:
        data: target ndarray of the original data
        gen_data: target ndarray of the generated data
    Return:
        None
    """

    N = len(gen_data)  # number of generated data

    # check shape of input data, raise error if it's not valid
    if len(data.shape) != 3 or data.shape[0] != 3:
        raise ValueError('shape of input data should be (3, 26, 26)')

    # check label of input data, raise error if it's not valid
    if label is None:
        title = 'No label provided'
    elif type(label) == int and label in LABEL2TEXT:
        title = LABEL2TEXT[label]
    else:
        raise ValueError('label of input data should between 0 and 8')

    bdry, dfct, nrml = data[0, :, :], data[1, :, :], data[2, :, :]

    plt.figure()
    plt.subplots_adjust(left=0.2, right=0.8, top=0.93, bottom=0.05, wspace=0, hspace=0.2)
    plt.suptitle(title, fontsize=14)
    plt.subplot(1 + N, 3, 1)
    plt.title('boundary')
    plt.imshow(bdry)
    plt.subplot(1 + N, 3, 2)
    plt.title('defect')
    plt.imshow(dfct)
    plt.subplot(1 + N, 3, 3)
    plt.title('normal')
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

def demo(path_to_data, path_to_generated_data, index_to_demo=None):
    """Display an image in the wafer dataset, and also the corresponding
    generated samples.

    Args:
        path_to_data: directory to data
        path_to_generated_data: directory to generated data
        index_to_demo (int): index of image in the dataset, if None then random
    Return:
        None
    """

    dataset = Dataset(path_to_data)
    gen_dataset = Dataset(path_to_generated_data, generated=True)

    if index_to_demo is None:
        # no index given, just random sample it
        index_to_demo = np.random.randint(0, len(dataset))
    elif type(index_to_demo) != int or not (0 <= index_to_demo < len(dataset)):
        raise ValueError('index should between 0 and {}'.format(len(dataset)))

    print('index to demo: {}'.format(index_to_demo))

    # read original data
    data, label = dataset[index_to_demo]

    # read generated data
    gen_data = []
    for i in range(5):
        gd, _ = gen_dataset[index_to_demo * 5 + i]
        gen_data.append(gd)
    # display original data and generated data
    display_image_with_gen_data(data, gen_data, int(label))


if __name__ == "__main__":
    demo(path_to_data='./wafer', path_to_generated_data='./output', index_to_demo=9)
