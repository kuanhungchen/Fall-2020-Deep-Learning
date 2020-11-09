import os
import numpy as np
import torch.utils.data

from torchvision.transforms import transforms


class WAFER_dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data):
        self.datas = np.load(os.path.join(path_to_data, "data.npy"))
        self.labels = np.load(os.path.join(path_to_data, "label.npy"))
    
    def __len__(self):
        return self.datas.shape[0]
    
    def __getitem__(self, index):
        """
        Args:
            index: integer
        Return:
            data.shape = (26, 26, 3)
            label.shape = (1)
        """
        data = self.datas[index]
        data = self.preprocess(data)
        data = data.view(3, 26, 26)

        label = self.labels[index]
        label = label.reshape(1)
        
        return data, label
    
    @staticmethod
    def preprocess(data):
        data = transforms.ToTensor()(data)
        # transform = transforms.Compose([
            # transforms.ToTensor()
        # ])
        # data = transform(data)
        return data

