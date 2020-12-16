import os
import cv2


class Fruit_dataset:
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
        label = self.labels[index]

        return image, label


if __name__ == '__main__':
    dataset = Fruit_dataset('./data', mode='train')
    print('len of dataset:', len(dataset))
    img, lbl = dataset[0]
    print('img shape:', img.shape)
    print('label:', lbl)
