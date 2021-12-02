from src.dataset import WAFER_dataset as Dataset

from matplotlib import pyplot as plt


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

def display_image(data, label=None):
    """Display single image (with three channels separated) given its ndarray
    by using matplotlib package.

    Args:
        data: target ndarray of the image which should contain three channels
        label (int): label of the input data
    Return:
        None
    """

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

    # draw the three components in a PIL window
    plt.figure()
    plt.suptitle(title, fontsize=14)
    plt.subplot(1, 3, 1)
    plt.title('boundary')
    plt.imshow(bdry)
    plt.subplot(1, 3, 2)
    plt.title('defect')
    plt.imshow(dfct)
    plt.subplot(1, 3, 3)
    plt.title('normal')
    plt.imshow(nrml)
    plt.show()


if __name__ == '__main__':
    dataset = Dataset(path_to_data='./wafer')
    for i in range(0, len(dataset)):
        # visualize data in the dataset
        data, label = dataset[i]
        if label != 8:
            continue
        print(i)
        # display_image(data, int(label))
        # break
