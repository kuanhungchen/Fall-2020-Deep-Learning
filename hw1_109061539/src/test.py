import os
import numpy as np

from src.dataset import MNIST_dataset as Dataset
from src.model import Model


def test(path_to_test_data, path_to_checkpoint):
    """
    Args:
        path_to_test_data: path to testing data
        path_to_checkpoint: path to trained model weights
    """
    
    test_dataset = Dataset(path_to_test_data, mode="test")
    print("load testing sample: {}".format(len(test_dataset)))

    model = Model()
    model.load(path_to_checkpoint)
    print("load model weights from {}".format(path_to_checkpoint))

    test_hit = 0
    for data_idx in np.arange(len(test_dataset)):
        res = test_dataset[data_idx]
        image, label = res["image"], res["label"]

        pred = model.predict(image)
        test_hit += int(pred) == int(label[0])

    print("Hit: {} | Miss: {} | Acc.: {}".format(test_hit, len(test_dataset) - test_hit, test_hit / len(test_dataset)))

if __name__ == "__main__":
    test("MNIST/test", "checkpoints/20201101_150637/model_70.npy")
