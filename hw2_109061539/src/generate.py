import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.demo import display_image
from src.model import AutoEncoder as AE
from src.dataset import WAFER_dataset as Dataset


def gaussian_noise(size, mean, variance):
    """Given size, mean and variange, return a matrix of Gaussian distribution
    Args:
        size (int): the width and height of gaussian noise matrix
        mean: mean of Gaussian distribution
        variance: variance of Gaussian distribution
    Return:
        noise: output gaussian noise matrix, shape = (size, size)
    """
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x * x + y * y)
    noise = np.exp(-((d - mean) ** 2 / (2 * variance)))

    return noise


def generate(path_to_data, path_to_generated_data, path_to_checkpoint):
    """Given model weights, generate new samples from the original dataset
    Args:
        path_to_data: directory to data
        path_to_generated_data: directory to save generated samples
        path_to_checkpoint: directory to target model weights
    Return:
        None
    """
    dataset = Dataset(path_to_data=path_to_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # create model
    model = AE()
    # load target model weights
    model.load(path_to_checkpoint)
    mode = model.float()
    
    # create array for generated samples
    generated_data = np.zeros((1, 26, 26, 3))
    generated_label = np.zeros((1, 1))

    # create gaussian noise matrix
    noise = gaussian_noise(5, 0, 1)
    
    save_flag = False
    for batch_idx, (data, label) in enumerate(dataloader):
        # if batch_idx < 500: continue

        display_image(data)
        data = data.float()
        
        # encode the data to latent
        latent = model.eval().encode(data)
        latent = latent.float()
        
        for _ in range(1):
            # add Gaussian noise to the latent
            noised_latent = latent
            
            for c in range(latent.shape[1]):
                for i in range(latent.shape[2]):
                    for j in range(latent.shape[3]):
                        noised_latent[0][c][i][j] += noise[i][j]
            
            # decode the noised latent back to reconstructed data
            reconstructed_data = model.eval().decode(noised_latent)

            reconstructed_data = reconstructed_data.detach().numpy()
            reconstructed_data = np.transpose(reconstructed_data, (0, 2, 3, 1))

            # convert reconstructed data to 0 or 1 per pixel
            for i in range(26):
                for j in range(26):
                    max_channel = np.argmax(reconstructed_data[0, i, j, :])
                    for c in range(3):
                        reconstructed_data[0, i, j, c] = 1 if c == max_channel else 0
            
            display_image(reconstructed_data)

            # append to array
            generated_data = np.concatenate((generated_data, reconstructed_data), axis=0)
            generated_label = np.concatenate((generated_label, label), axis=0)

        if batch_idx == 20:
            break
    
    generated_data = generated_data[1:]
    generated_label = generated_label[1:]
    
    if save_flag:
        np.save(os.path.join(path_to_generated_data, 'gen_data'), generated_data)
        np.save(os.path.join(path_to_generated_data, 'gen_label'), generated_label)
 

if __name__ == '__main__':
    generate(path_to_data='./wafer', path_to_generated_data='./output', path_to_checkpoint='./checkpoints/tmp/model_last.pth')
