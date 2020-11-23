import os
import torch
import numpy as np

from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        """Autoencoder network for generating wafer data
        """
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=0),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=7, padding=0),
            nn.ReLU()
        )

    def encode(self, data):
        """
        Args:
            data: input data, shape = (batch_size, 3, 26, 26)

        Return:
            latents: output encoded latent, shape = (batch_size, 8, 12, 12)
        """

        latent = self.encoder(data)

        return latent

    def decode(self, latent):
        """
        Args:
            latents: input latent, shape = (batch_size, 8, 12, 12)

        Return:
            data: output decoded data, shape = (batch_size, 3, 26, 26)
        """

        data = self.decoder(latent)

        return data

    def loss(self, data, reconstructed_data):
        """
        Args:
            data: original data, shape = (batch_size, 3, 26, 26)
            reconstructed_data: data from decoder, shape = (batch_size, 3, 26, 26)
        Return:
            loss.shape = ()
        """
        
        loss_function = nn.MSELoss()
        loss = loss_function(data, reconstructed_data)

        return loss
    
    def save(self, path_to_checkpoints_dir, tag):
        """
        Args:
            path_to_checkpoints_dir: directory to save model weights
            tag: tag about current model weights
        Return:
            None
        """
        filename = os.path.join(path_to_checkpoints_dir, "model_{}.pth".format(str(tag)))
        torch.save(self.state_dict(), filename)

    def load(self, path_to_checkpoint):
        """
        Args:
            path_to_checkpoint: path to target model weights
        Return:
            None
        """
        self.load_state_dict(torch.load(path_to_checkpoint))
