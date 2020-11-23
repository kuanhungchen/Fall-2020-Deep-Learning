import datetime
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import WAFER_dataset as Dataset
from model import AutoEncoder as AE


def train(path_to_data, path_to_checkpoints):
    """Train a model from scratch
    Args:
        path_to_data: directory to data
        path_to_checkpoints: directory to save model weights
    Return:
        None
    """
    dataset = Dataset(path_to_data=path_to_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # create model
    model = AE()
    model = model.float()
    model = model.cuda()

    # create optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epoch_num = 50000
    for epoch_idx in range(1, epoch_num + 1):
        total_loss = []
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.float()
            data = data.cuda()
            
            # encode the data to latent
            latent = model.train().encode(data)

            # decode the latent back to reconstructed data
            reconstructed_data = model.train().decode(latent)

            # compute loss
            loss = model.loss(data, reconstructed_data)
            
            total_loss.append(loss)
            optimizer.zero_grad()

            # backward
            loss.backward()

            # update model weights
            optimizer.step()

        # print training details
        print("[{}] Epoch: {:2d} | Loss: {:.4f}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch_idx, sum(total_loss) / len(total_loss)))
        
        if epoch_idx % 10000 == 0:
            # save current model weights
            model.save(path_to_checkpoints, tag=str(epoch_idx))

    model.save(path_to_checkpoints, tag='last')


if __name__ == "__main__":
    train(path_to_data="./wafer", path_to_checkpoints="./checkpoints")
