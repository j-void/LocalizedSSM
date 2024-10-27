from .unet3d_new import LocalizationModule, UNetModule
from .deepssm import DeepSSMNetModified
import torch
import math
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from util.net_utils import compose_matrix
from .extras import GaussianSmoothing

def weight_init(module, initf):
	"""
	Applies a specified initialization function to all weights within a module
	"""
	def foo(m):
		classname = m.__class__.__name__.lower()
		if isinstance(m, module):
			initf(m.weight)
	return foo

class Image2Shape(nn.Module):
    def __init__(self, in_channels, out_channels, cfg, train_data, f_maps=64, num_levels=4, fine_tune=False):
        """
        Initialize the Image2Shape model

        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
            cfg (Config): The configuration object
            train_data (TrainDataset): The training dataset
            f_maps (int, optional): The number of feature maps in the UNet. Defaults to 64.
            num_levels (int, optional): The number of levels in the UNet. Defaults to 4.
            fine_tune (bool, optional): Whether to fine tune the model. Defaults to False.
        """
        super(Image2Shape, self).__init__()
        
        # Set the device to the specified device
        self.device = cfg.DEVICE
        
        # Set the crop dimensions
        self.crop_dims = cfg.crop_dims
        
        # Set the save directory
        self.save_dir = os.path.join(cfg.checkpoint_dir, "save")
        
        # Make the save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
                        
        # Initialize the UNetModule
        self.snet = UNetModule(in_channels, out_channels, cfg.img_dims, f_maps=f_maps, num_levels=num_levels)
        self.snet.to(self.device)
        # Initialize the LocalizationModule
        self.lnet = LocalizationModule(in_channels, out_channels, cfg.crop_dims, f_maps=16, num_levels=5)
        self.lnet.to(self.device)
        
        # Initialize some variables
        self.num_corr = train_data.num_corr
        mean_score = train_data.mean_score
        std_score = train_data.std_score
        
        # Initialize the DeepSSMNetModified
        self.dnet = DeepSSMNetModified(train_data.num_latent, cfg.crop_dims, self.num_corr, mean_score, std_score)
        self.dnet.to(self.device)
        
        # Initialize the number of PCA vectors
        num_pca = train_data.num_pca
        
        # Initialize the weights
        self.snet.apply(weight_init(module=nn.Conv3d, initf=nn.init.kaiming_uniform_))	
        self.snet.apply(weight_init(module=nn.Linear, initf=nn.init.kaiming_uniform_))
        
        self.lnet.apply(weight_init(module=nn.Conv3d, initf=nn.init.kaiming_uniform_))	
        self.lnet.apply(weight_init(module=nn.Linear, initf=nn.init.kaiming_uniform_))
        
        self.dnet.apply(weight_init(module=nn.Conv3d, initf=nn.init.kaiming_uniform_))	
        self.dnet.apply(weight_init(module=nn.Linear, initf=nn.init.kaiming_uniform_))
        
        # Initialize the weights for the LocalizationModule, start with no traslation and rotation
        self.lnet.fc_loc[2].weight.data.zero_()
        self.lnet.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
        
        # Gaussian smoothing
        self.smoothing = GaussianSmoothing(1, 5, 2)
        self.smoothing.to(self.device)
             
        
        ## fix the last layer of deepSSMNet to act like PCA method
        if fine_tune == False:
            orig_mean = np.loadtxt(os.path.join(cfg.pca_info_dir, 'mean.particles'))
            orig_pc = np.zeros([num_pca, self.num_corr*3])
            for i in range(num_pca):
                temp = np.loadtxt(os.path.join(cfg.pca_info_dir ,'pcamode' + str(i) + '.particles'))
                orig_pc[i, :] = temp.flatten()
            bias = torch.from_numpy(orig_mean.flatten()).to(self.device) # load the mean 
            weight = torch.from_numpy(orig_pc.T).to(self.device) # load the PCA vectors 
            self.dnet.decoder.fc_fine.bias.data.copy_(bias)
            self.dnet.decoder.fc_fine.weight.data.copy_(weight)
            ### for the initial steps set the gradient of the final layer to be zero
            for param in self.dnet.decoder.fc_fine.parameters():
                param.requires_grad = False

        # Define the optimizer, parameters and scheduler
        train_params = list(self.lnet.parameters()) + list(self.dnet.parameters()) + list(self.snet.parameters())
        self.optimizer = torch.optim.Adam(train_params, cfg.learning_rate, betas=(0.9, 0.999))
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.99)
        self.optimizer.zero_grad()
        
        
        ## define loss functions
        self.criterion_seg = nn.BCELoss()
        self.criterion_points = nn.MSELoss()
        self.criterion_sim = nn.MSELoss()
        self.criterion_lpoints = nn.MSELoss()
        
    def MSE(self, predicted, ground_truth):
        """
        Calculates the Mean Squared Error between predicted and ground truth.
        """
        return torch.mean((predicted - ground_truth)**2)
        
    def focal_loss(self, pred, ground_truth, a = 0, c = 0):
        '''
        Emplioys general formulation of focal loss, the pred and groundtruth can either be PCA/latent space or the correspondences directly.
        a = 0 will results in standard MSE loss
        '''
        l = torch.abs(pred - ground_truth)
        out = l**2 / (1 + torch.exp(a*(c - l)))
        return torch.mean(out)
    
    def focal_rel_loss(self, pred: torch.Tensor, ground_truth: torch.Tensor, mean: torch.Tensor, a: float = 0, c: float = 0) -> torch.Tensor:
        """
        Calculate the relative focal loss between predicted and ground truth, normalized by the focal loss between mean and ground truth.

        Args:
            pred (torch.Tensor): Predicted tensor
            ground_truth (torch.Tensor): Ground truth tensor
            mean (torch.Tensor): Mean tensor
            a (float, optional): Focal loss parameter. Defaults to 0.
            c (float, optional): Focal loss parameter. Defaults to 0.

        Returns:
            torch.Tensor: Relative focal loss
        """
        return self.focal_loss(pred, ground_truth, a, c) / self.focal_loss(mean, ground_truth, a, c)

        
    def forward(self, img, seg, points_world, points_local, mshape, ref):
        """
        Forward pass through the full network.

        Args:
            img (torch.Tensor): The input image
            seg (torch.Tensor): The segmentation of the input image
            points_world (torch.Tensor): The world coordinates of the points
            points_local (torch.Tensor): The local coordinates of the points
            mshape (torch.Tensor): The shape of the mesh
            ref (torch.Tensor): The reference mesh

        Returns:
            losses (list): A list of the loss values
            outputs (list): A list of the output values
        """
        # predict the segmentation
        pred_seg = self.snet(img)
        # calculate the segmentation loss
        loss_seg = self.criterion_seg(pred_seg, seg)

        with torch.no_grad():
            # extract the region of interest based on segmentation output
            crop_img = torch.zeros([img.shape[0], img.shape[1], self.crop_dims[2], self.crop_dims[1], self.crop_dims[0]]).to(self.device)
            crop_seg = torch.zeros([seg.shape[0], seg.shape[1], self.crop_dims[2], self.crop_dims[1], self.crop_dims[0]]).to(self.device)

            # Find the indices of the points in the segmentation that are greater than 0.9
            crop_idx = torch.where(pred_seg>0.9)
            x_min = y_min = z_min = 0
            max_dt = 0
            
            if crop_idx[4].nelement() != 0:
                # Get the minimum and maximum of the x, y and z coordinates of the points
                x_min = crop_idx[4].min()
                x_max = crop_idx[4].max()
                y_min = crop_idx[3].min()
                y_max = crop_idx[3].max()
                z_min = crop_idx[2].min()
                z_max = crop_idx[2].max()
                # Crop the image based on the minimum and maximum of the x, y and z coordinates
                crop_img = img[:,:,z_min:z_max,y_min:y_max,x_min:x_max]
                # Get the maximum of the x, y and z dimensions of the cropped image
                max_dt = max(x_max-x_min, y_max-y_min, z_max-z_min)
                # Calculate the padding for the x, y and z dimensions of the cropped image
                xpad = int(abs(max_dt/2 - (x_max-x_min)/2))
                ypad = int(abs(max_dt/2)- (y_max-y_min)/2)
                zpad = int(abs(max_dt/2)- (z_max-z_min)/2)
                
                
                # Pad the cropped image and segmentation to the desired dimensions
                if crop_img.shape[4] > 0 and crop_img.shape[3] >  0 and crop_img.shape[2] > 0:
                    crop_img = F.pad(crop_img, (zpad, zpad, ypad, ypad, xpad, xpad), "constant", 0)
                    # Get the maximum of the x, y and z dimensions of the cropped image
                    max_s = max(crop_img.shape[4], crop_img.shape[3], crop_img.shape[2])
                    # Pad the cropped image to the maximum of the x, y and z dimensions
                    crop_img = F.pad(crop_img, (0, max_s-crop_img.shape[4], 0, max_s-crop_img.shape[3], 0, max_s-crop_img.shape[2]), "constant", 0)
                    # Interpolate the cropped image to the desired dimensions
                    crop_img = F.interpolate(crop_img, self.crop_dims, mode='trilinear')
                else:
                    # Create a tensor with zeros if the cropped image is empty
                    crop_img = torch.zeros([img.shape[0], img.shape[1], self.crop_dims[2], self.crop_dims[1], self.crop_dims[0]]).to(self.device)
                
                # Crop the segmentation to the desired dimensions
                crop_seg = pred_seg[:,:, z_min:z_max,y_min:y_max,x_min:x_max]
                
                # Do the same for the segmentation
                if crop_seg.shape[4] > 0 and crop_seg.shape[3] >  0 and crop_seg.shape[2] > 0:
                    crop_seg = F.pad(crop_seg, (zpad, zpad, ypad, ypad, xpad, xpad), "constant", 0)
                    max_s = max(crop_seg.shape[4], crop_seg.shape[3], crop_seg.shape[2])
                    crop_seg = F.pad(crop_seg, (0, max_s-crop_seg.shape[4], 0, max_s-crop_seg.shape[3], 0, max_s-crop_seg.shape[2]), "constant", 0)
                    crop_seg = F.interpolate(crop_seg, self.crop_dims, mode='trilinear')
                else:
                    crop_seg = torch.zeros([seg.shape[0], seg.shape[1], self.crop_dims[2], self.crop_dims[1], self.crop_dims[0]]).to(self.device)        
        
        # Use Spatial Transformer Network to align the segmentation and the image
        stn_seg, stn_img, theta = self.lnet(crop_img, mshape, crop_seg)
        
        # Calculate the similarity  / registration loss
        loss_sim = self.criterion_sim(stn_seg, mshape)
        
        with torch.no_grad():
            # Apply Gaussian Smoothing the output of the segmentation output from the Spatial Transformer Network
            crop_reg = self.smoothing(stn_seg)
            # Pad the smoothed output to the original size and combine with the registered image, this act as an attention map
            crop_reg = F.pad(crop_reg, (2, 2, 2, 2, 2, 2)) * torch.max(stn_img) + stn_img
        
        # Predict the points in the world coordinate system
        pred_pca, pred_points_w = self.dnet(crop_reg)
        
        # Calculate the points loss
        loss_points_w = self.criterion_points(pred_points_w, points_world)

        
        
        return [loss_seg, loss_sim, loss_points_w], [pred_seg, pred_points_w, crop_img, crop_seg, stn_img, stn_seg ]

    

    def val(self, img, seg, mshape, ref):
        """
        Validation function
        """
        # predict the segmentation
        pred_seg = self.snet(img)
        
        with torch.no_grad():
            # extract the region of interest based on segmentation output - Same as training
            crop_img = torch.zeros([img.shape[0], img.shape[1], self.crop_dims[2], self.crop_dims[1], self.crop_dims[0]]).to(self.device)
            crop_seg = torch.zeros([seg.shape[0], seg.shape[1], self.crop_dims[2], self.crop_dims[1], self.crop_dims[0]]).to(self.device)

            crop_idx = torch.where(pred_seg>0.9)
            x_min = y_min = z_min = 0
            max_dt = 0

            if crop_idx[4].nelement() != 0:
                # Get the bounding box of the foreground pixels
                x_min = crop_idx[4].min()
                x_max = crop_idx[4].max()
                y_min = crop_idx[3].min()
                y_max = crop_idx[3].max()
                z_min = crop_idx[2].min()
                z_max = crop_idx[2].max()
                
                # Crop the image and the segmentation output to the bounding box
                crop_img = img[:,:,z_min:z_max,y_min:y_max,x_min:x_max]
                max_dt = max(x_max-x_min, y_max-y_min, z_max-z_min)
                xpad = int(abs((x_max-x_min)/2 - max_dt/2))
                ypad = int(abs((y_max-y_min)/2 - max_dt/2))
                zpad = int(abs((z_max-z_min)/2 - max_dt/2))
                
                # Pad the cropped image and segmentation output to the desired size
                if crop_img.shape[4] > 0 and crop_img.shape[3] >  0 and crop_img.shape[2] > 0:
                    crop_img = F.pad(crop_img, (zpad, zpad, ypad, ypad, xpad, xpad), "constant", 0)
                    max_s = max(crop_img.shape[4], crop_img.shape[3], crop_img.shape[2])
                    crop_img = F.pad(crop_img, (0, max_s-crop_img.shape[4], 0, max_s-crop_img.shape[3], 0, max_s-crop_img.shape[2]), "constant", 0)
                    crop_img = F.interpolate(crop_img, self.crop_dims, mode='trilinear')
                else:
                    crop_img = torch.zeros([img.shape[0], img.shape[1], self.crop_dims[2], self.crop_dims[1], self.crop_dims[0]]).to(self.device)
                
                crop_seg = pred_seg[:,:, z_min:z_max,y_min:y_max,x_min:x_max]#.detach()
                

                if crop_seg.shape[4] > 0 and crop_seg.shape[3] >  0 and crop_seg.shape[2] > 0:
                    crop_seg = F.pad(crop_seg, (zpad, zpad, ypad, ypad, xpad, xpad), "constant", 0)
                    max_s = max(crop_seg.shape[4], crop_seg.shape[3], crop_seg.shape[2])
                    crop_seg = F.pad(crop_seg, (0, max_s-crop_seg.shape[4], 0, max_s-crop_seg.shape[3], 0, max_s-crop_seg.shape[2]), "constant", 0)
                    crop_seg = F.interpolate(crop_seg, self.crop_dims, mode='trilinear')
                else:
                    crop_seg = torch.zeros([seg.shape[0], seg.shape[1], self.crop_dims[2], self.crop_dims[1], self.crop_dims[0]]).to(self.device)
        
        
        # Use Spatial Transformer Network to align the segmentation and the image
        stn_seg, stn_img, theta = self.lnet(crop_img, mshape, crop_seg)
        
        with torch.no_grad():
            # Apply Gaussian Smoothing the output of the segmentation output from the Spatial Transformer Network
            crop_reg = self.smoothing(stn_seg)
            # Pad the smoothed output to the original size and combine with the registered image, this act as an attention map
            crop_reg = F.pad(crop_reg, (2, 2, 2, 2, 2, 2)) * torch.max(stn_img) + stn_img
        
        # Predict the points in the world coordinate system
        pred_pca, pred_points_w = self.dnet(crop_reg)
        
        
        return [pred_seg, pred_points_w, stn_seg, stn_img]
    
    
    def save_network(self, name):
        """
        Saves the network's state dictionary to the checkpoint directory
        Args:
            name: The name of the model to save
        """
        torch.save(self.snet.state_dict(), os.path.join(self.save_dir, "snet_"+name+".torch"))
        torch.save(self.lnet.state_dict(), os.path.join(self.save_dir, "lnet_"+name+".torch"))
        torch.save(self.dnet.state_dict(), os.path.join(self.save_dir, "dnet_"+name+".torch"))
    
    def load_pretrained(self, name):
        """
        Loads the pre-trained weights of the network
        Args:
            name: The name of the pre-trained model to load
        """
        self.snet.load_state_dict(torch.load(os.path.join(self.save_dir, "snet_"+name+".torch"), map_location=self.device))
        self.lnet.load_state_dict(torch.load(os.path.join(self.save_dir, "lnet_"+name+".torch"), map_location=self.device))
        self.dnet.load_state_dict(torch.load(os.path.join(self.save_dir, "dnet_"+name+".torch"), map_location=self.device))
    
