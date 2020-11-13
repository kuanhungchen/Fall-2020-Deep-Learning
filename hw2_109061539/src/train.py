import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import WAFER_dataset as Dataset
from model import AutoEncoder as AE


def train(path_to_data, path_to_checkpoints=""):

    dataset = Dataset(path_to_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AE()
    model = model.float()
    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    
    epoch_num = 1000
    for epoch_idx in range(1, epoch_num + 1):
        total_loss = []
        for batch_idx, (data, label) in enumerate(dataloader):
            data = data.float()
            data = data.cuda()

            label = label.float()
            label = label.cuda()

            latent = model.train().encode(data)
            latent = latent.float()

            # print("latent.shape ==>", latent.shape)

            reconstructed_data = model.train().decode(latent)
            reconstructed_data = reconstructed_data.float()

            # print("reconstructed_data.shape ==>", reconstructed_data.shape)
            
            loss = model.loss(data, reconstructed_data)
            
            total_loss.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: {:2d} | Loss: {:.4f}".format(epoch_idx, sum(total_loss) / len(total_loss)))


if __name__ == "__main__":
    train(path_to_data="wafer")

