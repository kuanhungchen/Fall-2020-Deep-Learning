import os
import numpy as np
import torch.utils.data

from torchvision.transforms import transforms


class WAFER_dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data, generated=False):
        """Wafer dataset
        """

        if generated:
            self.datas = np.load(os.path.join(path_to_data, 'gen_data.npy'))
            self.labels = np.load(os.path.join(path_to_data, 'gen_label.npy'))
        else:
            self.datas = np.load(os.path.join(path_to_data, 'data.npy'))
            self.labels = np.load(os.path.join(path_to_data, 'label.npy'))
    
    def __len__(self):
        """Return the length of dataset.
        Args:
            None
        Return:
            l (int): length of dataset
        """

        l = self.datas.shape[0]
        return l
    
    def __getitem__(self, index):
        """Get item from dataset given an index.
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
        """Convert the data into tensor.
        Args:
            data: target data
        Return:
            data: tensor of the input data
        """

        data = transforms.ToTensor()(data)
        return data


if __name__ == '__main__':
    dataset = WAFER_dataset(path_to_data='./wafer', generated=False)
    i, l = dataset[0]
    print(type(i))
    print(i.shape)
