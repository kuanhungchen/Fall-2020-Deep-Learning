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
                    img = cv2.imread(os.path.join(self.path_to_data, text, filename), 0)
                    self.images.append(img)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        image = np.array(image, dtype=float)
        image = np.reshape(image, (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNEL))
        image /= 255.0

        label = self.labels[index]
        label = np.array(label, dtype=int)
        label = np.reshape(label, (1, -1))

        return image, label


if __name__ == '__main__':
    dataset = Dataset('./data', mode='train')
    print('Length of dataset:', len(dataset))
    img, lbl = dataset[0]
    print('Image:')
    print('  - shape:', img.shape)
    print('  - type:', type(img))
    print('Label:')
    print('  - shape:', lbl.shape)
    print('  - type:', type(lbl))
