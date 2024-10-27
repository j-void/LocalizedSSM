
import torch
from torch import nn
import sys
import json
import numpy as np
from collections import OrderedDict
import os

def unwhiten_PCA_scores(pca_load, mean_score, std_score, device):
	"""
	Unwhitens the PCA scores to get the original scores.
	
	Args:
		pca_load: The PCA scores to unwhiten.
		mean_score: The mean PCA score.
		std_score: The standard deviation PCA score.
		device: The device to use.

	Returns:
		The unwhitened PCA scores.
	"""
	mean_score = torch.tensor(mean_score).to(device).float()
	std_score = torch.tensor(std_score).to(device).float()
	mean_score = mean_score.unsqueeze(0).repeat(pca_load.shape[0], 1)
	std_score = std_score.unsqueeze(0).repeat(pca_load.shape[0], 1)
	pca_new = pca_load*(std_score) + mean_score
	return pca_new

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

def poolOutDim(inDim, kernel_size, padding=0, stride=0, dilation=1):
    """
    Calculates the output dimension of a pooling layer.
    """
    # If stride is 0, set it to the kernel size
    if stride == 0:
        stride = kernel_size

    # Calculate the output dimension
    num = inDim + 2*padding - dilation*(kernel_size - 1) - 1
    outDim = int(np.floor(num/stride + 1))

    return outDim

    

class ConvolutionalBackbone(nn.Module):
	def __init__(self,latent_dim, num_latent):
		super(ConvolutionalBackbone, self).__init__()
		
		self.features = nn.Sequential(
			nn.Conv3d(1, 12, 5),
			nn.BatchNorm3d(12),
			nn.PReLU(),
			nn.MaxPool3d(2),

			nn.Conv3d(12, 24, 5),
			nn.BatchNorm3d(24),
			nn.PReLU(),
			nn.Conv3d(24, 48, 5),
			nn.BatchNorm3d(48),
			nn.PReLU(),
			nn.MaxPool3d(2),

			nn.Conv3d(48, 96, 5),
			nn.BatchNorm3d(96),
			nn.PReLU(),
			nn.Conv3d(96, 192, 5),
			nn.BatchNorm3d(192),
			nn.PReLU(),
			nn.MaxPool3d(2),

            Flatten(),

			nn.Linear(latent_dim, 384),
			nn.PReLU(),
			nn.Linear(384, 96),
			nn.PReLU(),
			nn.Linear(96, num_latent)
		)

	def forward(self, x):
		#print(x.shape)
		x_features = self.features(x)
		#print(x.shape)
		return x_features


class DeterministicLinearDecoder(nn.Module):
	def __init__(self, num_latent, num_corr):
		super(DeterministicLinearDecoder, self).__init__()
		self.num_latent = num_latent
		self.numL = num_corr
		self.fc_fine = nn.Linear(self.num_latent, self.numL*3)

	def forward(self, pca_load_unwhiten):
		corr_out = self.fc_fine(pca_load_unwhiten).reshape(-1, self.numL, 3)
		return corr_out


class CombinedEncoder(nn.Module):
	def __init__(self, num_latent, latent_dim , mean_score, std_score):
		super(CombinedEncoder, self).__init__()

		self.mean_score = mean_score
		self.std_score = std_score

		self.image_encoder = ConvolutionalBackbone(latent_dim, num_latent)

	def forward(self, image):
		pca_load = self.image_encoder(image)
		pca_load_unwhiten = unwhiten_PCA_scores(pca_load, self.mean_score, self.std_score, "cuda")
		return pca_load, pca_load_unwhiten

class DeepSSMNetModified(nn.Module):
    """
    The DeepSSMNetModified network is a modified version of the original DeepSSMNet network.
    """
    def __init__(self, num_latent, crop_dims, num_corr, mean_score, std_score):
        """
        Initializes the network with the given parameters.
        :param num_latent: The number of latent variables.
        :param crop_dims: The dimensions of the cropped input images.
        :param num_corr: The number of correspondence points.
        :param mean_score: The mean score of the PCA scores.
        :param std_score: The standard deviation of the PCA scores.
        """
        super(DeepSSMNetModified, self).__init__()
        self.num_latent = num_latent
        self.num_corr = num_corr
        self.out_fc_dim = np.copy(crop_dims)
        padvals = [4, 8, 8]
        for i in range(3):
            self.out_fc_dim[0] = poolOutDim(self.out_fc_dim[0] - padvals[i], 2)
            self.out_fc_dim[1] = poolOutDim(self.out_fc_dim[1] - padvals[i], 2)
            self.out_fc_dim[2] = poolOutDim(self.out_fc_dim[2] - padvals[i], 2)
        # Calculate the size of the latent space
        self.latent_size = self.out_fc_dim[0]*self.out_fc_dim[1]*self.out_fc_dim[2]*192
        self.encoder = CombinedEncoder(self.num_latent, self.latent_size, mean_score, std_score)
        self.decoder = DeterministicLinearDecoder(self.num_latent, self.num_corr)

    def forward(self, x):
        """
        Forward pass of the network.
        :param x: The input image.
        :return: The output of the network.
        """
        pca_load, pca_load_unwhiten = self.encoder(x)
        corr_out = self.decoder(pca_load_unwhiten)
        return [pca_load, corr_out]


## For debug purpose only   
if __name__ == "__main__":
	import nrrd
	DEVICE = "cuda"
	encoder = DeepSSMNetModified(16, [50, 50, 50], 23, np.ones((16)),np.ones((16))).to(DEVICE)
	float_tensor = torch.tensor(np.zeros((50, 50, 50))).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
	pose = torch.tensor(np.zeros((10))).float().unsqueeze(0).to(DEVICE)
	print(float_tensor.shape)
	output = encoder(float_tensor, pose)
	print(output[0].shape, output[1].shape)
