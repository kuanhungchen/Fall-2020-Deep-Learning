import argparse
import numpy as np

from src.dataset import Dataset
from src.model import Model
from src.utils import print_confusion_matrix



def test(path_to_data, path_to_checkpoint):
    # create dataset
    test_dataset = Dataset(path_to_data=path_to_data, mode='test')
    print('[Test] load testing sample: {}'.format(len(test_dataset)))

    # load model
    model = Model()
    print('[Test] model initialize successfully')
    model.load(path_to_checkpoint=path_to_checkpoint)
    print('[Test] model weights load successfully')

    # compute testing accuracy and confusion matrix
    testing_hit, testing_miss = 0, 0
    confusion_matrix_data = {
        'actual_0': [0, 0, 0],
        'actual_1': [0, 0, 0],
        'actual_2': [0, 0, 0]
    }
    for data_idx in range(len(test_dataset)):
        images, labels = test_dataset[data_idx]
        logits = model.forward(images)
        testing_hit += int(np.argmax(logits[0, :])) == int(labels[0, 0])
        testing_miss += int(np.argmax(logits[0, :])) != int(labels[0, 0])
        confusion_matrix_data['actual_' + str(labels[0, 0])][int(np.argmax(logits[0, :]))] += 1

    print('[Test] Testing accuracy: {:.4f}'.format(
        testing_hit / (testing_hit + testing_miss)))
    print('[Test] Confusion matrix:')
    print(print_confusion_matrix(confusion_matrix_data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data_dir', default='./data', help='path to data directory')
    parser.add_argument('-c', '--checkpoint', required=True, help='path to model checkpoint')
    args = parser.parse_args()

    path_to_data = args.data_dir
    path_to_checkpoint = args.checkpoint
    
    print('Argument:')
    print('  - path to data: {}'.format(path_to_data))
    print('  - path to checkpoint: {}'.format(path_to_checkpoint))
    test(path_to_data=path_to_data,
         path_to_checkpoint=path_to_checkpoint)
