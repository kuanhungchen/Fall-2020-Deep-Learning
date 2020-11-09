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
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, 3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2)
        )
        self.nrml_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, 3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2)
        )
        self.dfct_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, 3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2)
        )
    
    def forward(self, data):
        """
        Args:
            data.shape = (batch_size, 3 26, 26)
        """

        bdry = data[:, 0, :, :].view(-1, 1, 26, 26)
        nrml = data[:, 1, :, :].view(-1, 1, 26, 26)
        dfct = data[:, 2, :, :].view(-1, 1, 26, 26)
        
        bdry_logits = self.bdry_encoder(bdry)
        nrml_logits = self.nrml_encoder(nrml)
        dfct_logits = self.dfct_encoder(dfct)
        
        logits = np.stack((bdry_logits, nrml_logits, dfct_logits), axis=1)
        return logits

