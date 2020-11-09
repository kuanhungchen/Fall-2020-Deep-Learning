import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import WAFER_dataset as Dataset
from model import AutoEncoder as AE


def train(path_to_data, path_to_checkpoints=""):

    dataset = Dataset(path_to_data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = AE()
    model = model.float()

    optimizer = optim.SGD(model.parameters(), lr=3e-4)

    for batch_idx, (datas, labels) in enumerate(dataloader):
        logits = model.train().forward(datas.float())
        print(logits.shape)
        
        break

if __name__ == "__main__":
    train(path_to_data="wafer")

