import os
import cv2
import numpy as np


class Dataset:
    
    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    IMG_CHANNEL = 1
    text2label = {'Carambula': 0, 'Lychee': 1, 'Pear': 2}
    
    def __init__(self, path_to_data, mode):
        assert mode in ['train', 'val', 'test'], 'mode should be train, val or test'
        self.mode = mode
        self.images = []
        self.labels = []
        
        if self.mode == 'train':
            self.path_to_data = os.path.join(path_to_data, 'Data_train')
            for text, label in self.text2label.items():
                for filename in os.listdir(os.path.join(self.path_to_data, text))[:343]:
                    img = cv2.imread(os.path.join(self.path_to_data, text, filename), 0)
                    self.images.append(img)
                    self.labels.append(label)
        elif self.mode == 'val':
            self.path_to_data = os.path.join(path_to_data, 'Data_train')
            for text, label in self.text2label.items():
                for filename in os.listdir(os.path.join(self.path_to_data, text))[343:]:
                    img = cv2.imread(os.path.join(self.path_to_data, text, filename), 0)
                    self.images.append(img)
                    self.labels.append(label)
        else:
            self.path_to_data = os.path.join(path_to_data, 'Data_test')
            for text, label in self.text2label.items():
                for filename in os.listdir(os.path.join(self.path_to_data, text)):
                    img = cv2.imread(os.path.join(
                        self.path_to_data, text, filename), 0)
                    self.images.append(img)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            indexes = [int_or_slice]
        elif isinstance(int_or_slice, slice):
            start = int_or_slice.start if int_or_slice.start else 0
            stop = int_or_slice.stop if int_or_slice.stop else len(self.images)
            step = int_or_slice.step if int_or_slice.step else 1
            indexes = list(iter(range(start, stop, step)))
        else:
            indexes = list(int_or_slice)

        num_of_fetch = len(indexes)
        images, labels = [], []
        for i, index in enumerate(indexes):
            image = self.images[index]
            image = np.array(image, dtype=float)
            image /= 255.0
            images.append(image)

            label = self.labels[index]
            labels.append(label)

        images = np.array(images, dtype=float)
        images = np.reshape(images, (
            num_of_fetch, self.IMG_HEIGHT,
            self.IMG_WIDTH, self.IMG_CHANNEL))
        labels = np.array(labels, dtype=int)
        labels = np.reshape(labels, (num_of_fetch, 1))

        return images, labels
