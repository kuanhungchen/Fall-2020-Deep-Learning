import torch
import numpy as np

from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.bdry_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 2)
        )
        self.nrml_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 2)
        )
        self.dfct_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 2)
        )

        self.bdry_decoder = nn.Sequential(
            # nn.UpsamplingNearest2d(scale_factor=2),
            # nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(8, 16, 4, stride=2),
            nn.ReLU(),
            # nn.UpsamplingNearest2d(scale_factor=2),
            # nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(16, 1, 4, stride=2),
            nn.ReLU(),
            # nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.nrml_decoder = nn.Sequential(
            # nn.UpsamplingNearest2d(scale_factor=2),
            # nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(8, 16, 4, stride=2),
            nn.ReLU(),
            # nn.UpsamplingNearest2d(scale_factor=2),
            # nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(16, 1, 4, stride=2),
            nn.ReLU(),
            # nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.dfct_decoder = nn.Sequential(
            # nn.UpsamplingNearest2d(scale_factor=2),
            # nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(8, 16, 4, stride=2),
            nn.ReLU(),
            # nn.UpsamplingNearest2d(scale_factor=2),
            # nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(16, 1, 4, stride=2),
            nn.ReLU(),
            # nn.UpsamplingNearest2d(scale_factor=2),
        )
    
    def encode(self, data):
        """
        Args:
            data.shape = (batch_size, 3 26, 26)

        Return:
            latents.shape = (batch_size, 24, 5, 5)
        """

        bdry = data[:, 0, :, :].view(-1, 1, 26, 26)
        nrml = data[:, 1, :, :].view(-1, 1, 26, 26)
        dfct = data[:, 2, :, :].view(-1, 1, 26, 26)
        
        bdry_latent = self.bdry_encoder(bdry)
        nrml_latent = self.nrml_encoder(nrml)
        dfct_latent = self.dfct_encoder(dfct)
        
        latent = torch.cat((bdry_latent, nrml_latent, dfct_latent), 1)


        return latent

    def decode(self, latent):
        """
        Args:
            latents.shape = (batch_size, 24, 5, 5)

        Return:
            data.shape = (batch_size, 3, 26, 26)
        """
        bdry_latent = latent[:, 0:8, :, :].view(-1, 8, 5, 5)
        nrml_latent = latent[:, 8:16, :, :].view(-1, 8, 5, 5)
        dfct_latent = latent[:, 16:24, :, :].view(-1, 8, 5, 5)

        bdry = self.bdry_decoder(bdry_latent)
        nrml = self.nrml_decoder(nrml_latent)
        dfct = self.dfct_decoder(dfct_latent)

        data = torch.cat((bdry, nrml, dfct), 1)
        
        return data

    def loss(self, data, reconstructed_data):
        """
        Args:
            data.shape = (batch_size, 3, 26, 26)
            reconstructed_data.shape = (batch_size, 3, 26, 26)
        Return:
            loss.shape = ()
        """
        
        loss_function = nn.MSELoss()
        loss = loss_function(data, reconstructed_data)

        return loss

