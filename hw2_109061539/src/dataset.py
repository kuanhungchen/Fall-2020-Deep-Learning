import os
import numpy as np
import torch.utils.data

from torchvision.transforms import transforms


class WAFER_dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data, generated=False):
        self.generated = generated
        if generated:
            self.datas = np.load(os.path.join(path_to_data, 'gen_data.npy'))
            self.labels = np.load(os.path.join(path_to_data, 'gen_label.npy'))
        else:
            self.datas = np.load(os.path.join(path_to_data, 'data.npy'))
            self.labels = np.load(os.path.join(path_to_data, 'label.npy'))
        print(self.datas.shape)
        print(self.labels.shape)
    
    def __len__(self):
        return self.datas.shape[0]
    
    def __getitem__(self, index):
        """
        Args:
            index (int): index of data in the dataset
        Return:
            data: target image
            label: target label
        """
        data = self.datas[index]
        label = self.labels[index]
        
        data = self.preprocess(data)
        data = data.view(3, 26, 26)
        label = label.reshape(1)
        
        return data, label
    
    @staticmethod
    def preprocess(data):
        data = transforms.ToTensor()(data)
        return data


if __name__ == '__main__':
    dataset = WAFER_dataset(path_to_data='./wafer')
    i, l = dataset[0]
    print(type(i))
    print(i.shape)
