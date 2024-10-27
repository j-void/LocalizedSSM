import numpy as np
import torch
import os
import glob
import nrrd

from torch.utils.data import Dataset
import util.net_utils as util
from torch.nn import functional as F


def get_particles(model_path):
    """
    Load particles from a file and convert them into a list of float numpy arrays.

    Args:
        model_path (str): The path to the particles file.

    Returns:
        list: A list of float numpy arrays, where each array represents a particle.
    """
    f = open(model_path, "r")
    data = []
    for line in f.readlines():
        points = line.split()
        points = [float(i) for i in points]
        data.append(points)
    return data

class TrainDataset(Dataset):

    def __init__(self, cfg):
        """
        Initialize the dataset.

        Args:
            cfg (config): The configuration object.
        """
        super(TrainDataset, self).__init__()
        self.img_dims = cfg.img_dims
        self.images = glob.glob(os.path.join(cfg.img_dir, "*"))
        self.images.sort()
        self.segmentations = glob.glob(os.path.join(cfg.segmentation_dir, "*"))
        self.segmentations.sort()
        self.properties = glob.glob(os.path.join(cfg.properties_dir, "*"))
        self.properties.sort()
        self.pca_scores, self.mean_score, self.std_score = util.whiten_PCA_scores(np.load(os.path.join(cfg.pca_info_dir, "original_PCA_scores.npy")))
        self.world_particles, self.local_particles  = self.load_particles(cfg.particles_dir)
        self.num_latent = int(np.load(os.path.join(cfg.pca_info_dir, "latent_dim.npy")))
        self.num_corr = self.local_particles[0].shape[0]
        self.num_pca = self.pca_scores[0].shape[0]
        np.savez(os.path.join(cfg.checkpoint_dir, "params.npz"), mean_score=self.mean_score,  std_score=self.std_score, num_latent=self.num_latent, num_corr=self.num_corr, num_pca=self.num_pca)
        self.mean_info, self.mean_shape, self.ref = self.get_mean_shape(cfg)

    def __len__(self):
        return len(self.images)

    def load_particles(self, particle_dir):
        """
        Load particles from the given directory. 

        Args:
            particle_dir (str): The directory where the particles are stored.

        Returns:
            list: A list of two numpy arrays, the first one is the world particles and the second one is the local particles.
        """
        world_particle_list = []
        local_particle_list = []
        for file in os.listdir(particle_dir):
            if "local" in file:
                local_particle_list.append(os.path.join(particle_dir, file))
            if "world" in file:
                world_particle_list.append(os.path.join(particle_dir, file))
        
        world_particle_list = sorted(world_particle_list)
        local_particle_list = sorted(local_particle_list)

        world_particles = []
        local_particles = []

        for i in range(len(world_particle_list)):
            world_particles.append(get_particles(world_particle_list[i]))
            local_particles.append(get_particles(local_particle_list[i]))

        return [np.array(world_particles), np.array(local_particles)]

    def __getitem__(self, idx):
        image, _ = nrrd.read(self.images[idx])
        seg, _ = nrrd.read(self.segmentations[idx]) 
        points_world = self.world_particles[idx] 
        points_local = self.local_particles[idx]
        matrix_trs = torch.eye(4).float()
        image = torch.tensor(image).float()
        seg = torch.tensor(seg).float()
        return image.unsqueeze(0), seg.unsqueeze(0),\
            torch.tensor(points_world).float(), torch.tensor(points_local).float(), matrix_trs,\
            self.mean_shape, self.ref
        
    def get_mean_shape(self, cfg):
        """
        Get the mean shape of the dataset.

        Args:
            cfg (Config): The configuration object.

        Returns:
            tuple: A tuple containing the mean shape and the information about the mean shape.
        """
        # Read the reference shape
        ref_, info_ = nrrd.read(cfg.mean_shape)
        ref_ = torch.tensor(ref_).float().unsqueeze(0).unsqueeze(0)

        # Compute the bounding box of the foreground pixels
        crop_idx = torch.where(ref_>0.9)
        x_min = crop_idx[4].min()
        x_max = crop_idx[4].max()
        y_min = crop_idx[3].min()
        y_max = crop_idx[3].max()
        z_min = crop_idx[2].min()
        z_max = crop_idx[2].max()

        # Crop the reference shape to the bounding box
        mshape = ref_[:,:,z_min:z_max,y_min:y_max,x_min:x_max]

        # Compute the maximum dimension of the bounding box
        max_dt = max(x_max-x_min, y_max-y_min, z_max-z_min)

        # Pad the cropped shape to the desired size
        xpad = int(abs((x_max-x_min)/2 - max_dt/2))
        ypad = int(abs((y_max-y_min)/2 - max_dt/2))
        zpad = int(abs((z_max-z_min)/2 - max_dt/2))
        mshape = F.pad(mshape, (zpad, zpad, ypad, ypad, xpad, xpad), "constant", 0)
        max_s = max(mshape.shape[4], mshape.shape[3], mshape.shape[2])
        mshape = F.pad(mshape, (0, max_s-mshape.shape[4], 0, max_s-mshape.shape[3], 0, max_s-mshape.shape[2]), "constant", 0)

        # Interpolate the padded shape to the desired size
        scale_factor=cfg.crop_dims[0]/max_dt
        mean_shape = F.interpolate(mshape, cfg.crop_dims, mode='trilinear').squeeze().unsqueeze(0)

        # Save the mean shape to a file
        nrrd.write(os.path.join(cfg.checkpoint_dir, "mean_shape.nrrd"), mean_shape.squeeze().numpy())

        # Get the origin of the mean shape
        origin =  np.array([x_min, y_min, z_min]) + info_['space origin']

        # Create a dictionary containing the information about the mean shape
        mean_info = {'scale_factor':scale_factor, 'origin':origin, 'space_origin':info_['space origin']}

        # Return the mean shape and the information about it
        return mean_info, mean_shape, ref_


    

class DeepSSMCDataset(Dataset):
    def __init__(self, cfg):
        """
        For training cropped version of DeepSSM

        Args:
            cfg (Config): The configuration object.
        """
        super(DeepSSMCDataset, self).__init__()
        self.img_dims = cfg.img_dims
        self.crop_dims = cfg.crop_dims
        self.images = glob.glob(os.path.join(cfg.img_dir, "*"))
        self.images.sort()
        self.segmentations = glob.glob(os.path.join(cfg.segmentation_dir, "*"))
        self.segmentations.sort()
        self.pca_scores, self.mean_score, self.std_score = util.whiten_PCA_scores(np.load(os.path.join(cfg.pca_info_dir, "original_PCA_scores.npy")))
        self.world_particles, self.local_particles  = self.load_particles(cfg.particles_dir)
        self.num_latent = int(np.load(os.path.join(cfg.pca_info_dir, "latent_dim.npy")))
        self.num_corr = self.local_particles[0].shape[0]
        self.num_pca = self.pca_scores[0].shape[0]
        np.savez(os.path.join(cfg.checkpoint_dir, "params.npz"), mean_score=self.mean_score,  std_score=self.std_score, num_latent=self.num_latent, num_corr=self.num_corr, num_pca=self.num_pca)
        
        # Load the crops of the dataset
        self.crops = []
        for i in range(len(self.images)):
            crp = self.extract_shape(i)
            self.crops.append(crp)
            

    def __len__(self):
        return len(self.images)

    def load_particles(self, particle_dir):
        world_particle_list = []
        local_particle_list = []
        for file in os.listdir(particle_dir):
            if "local" in file:
                local_particle_list.append(os.path.join(particle_dir, file))
            if "world" in file:
                world_particle_list.append(os.path.join(particle_dir, file))
        
        world_particle_list = sorted(world_particle_list)
        local_particle_list = sorted(local_particle_list)

        world_particles = []
        local_particles = []

        for i in range(len(world_particle_list)):
            world_particles.append(get_particles(world_particle_list[i]))
            local_particles.append(get_particles(local_particle_list[i]))

        return [np.array(world_particles), np.array(local_particles)]

    def __getitem__(self, idx):
        points_world = self.world_particles[idx] 
        points_local = self.local_particles[idx]
        return self.crops[idx].float().unsqueeze(0), torch.tensor(points_world).float(), torch.tensor(points_local).float()
        
    def extract_shape(self, idx):
        """
        Extract the region of interest from the segmentation and image.

        Args:
            idx (int): The index of the image and segmentation to use.

        Returns:
            torch.Tensor: The cropped image.
        """

        # Read the segmentation and image
        seg_, _ = nrrd.read(self.segmentations[idx])
        image, _ = nrrd.read(self.images[idx])

        # Convert to tensors
        seg_ = torch.tensor(seg_).float().unsqueeze(0).unsqueeze(0)
        image = torch.tensor(image).float().unsqueeze(0).unsqueeze(0)

        # Find the indices of the points in the segmentation that are greater than 0.9
        crop_idx = torch.where(seg_>0.9)

        # Get the minimum and maximum of the x, y and z coordinates of the points
        x_min = crop_idx[4].min()
        x_max = crop_idx[4].max()
        y_min = crop_idx[3].min()
        y_max = crop_idx[3].max()
        z_min = crop_idx[2].min()
        z_max = crop_idx[2].max()

        # Crop the image based on the minimum and maximum of the x, y and z coordinates
        crop_img = image[:,:,z_min:z_max,y_min:y_max,x_min:x_max]

        # Get the maximum of the x, y and z dimensions of the cropped image
        max_dt = max(x_max-x_min, y_max-y_min, z_max-z_min)

        # Calculate the padding for the x, y and z dimensions of the cropped image
        xpad = int(abs(max_dt/2 - (x_max-x_min)/2))
        ypad = int(abs(max_dt/2)- (y_max-y_min)/2)
        zpad = int(abs(max_dt/2)- (z_max-z_min)/2)

        # Pad the cropped image
        if crop_img.shape[4] > 0 and crop_img.shape[3] >  0 and crop_img.shape[2] > 0:
            crop_img = F.pad(crop_img, (zpad, zpad, ypad, ypad, xpad, xpad), "constant", 0)
            max_s = max(crop_img.shape[4], crop_img.shape[3], crop_img.shape[2])
            crop_img = F.pad(crop_img, (0, max_s-crop_img.shape[4], 0, max_s-crop_img.shape[3], 0, max_s-crop_img.shape[2]), "constant", 0)
            # Interpolate the padded image to the desired size
            crop_img = F.interpolate(crop_img, self.crop_dims, mode='trilinear')
        else:
            crop_img = torch.zeros([crop_img.shape[0], crop_img.shape[1], self.crop_dims[2], self.crop_dims[1], self.crop_dims[0]])

        return crop_img.squeeze()
    

class DeepSSMFDataset(Dataset):
    def __init__(self, cfg):
        """
        Training DeepSSM with full images

        Args:
            cfg (config): The configuration object.
        """
        super(DeepSSMFDataset, self).__init__()
        # Only done for left atrium example: set the image dimensions to [64, 160, 192]
        cfg.img_dims = [64, 160, 192]#[125, 120, 166]#
        self.img_dims = cfg.img_dims
        self.images = glob.glob(os.path.join(cfg.img_dir, "*"))
        self.images.sort()
        self.pca_scores, self.mean_score, self.std_score = util.whiten_PCA_scores(
            np.load(os.path.join(cfg.pca_info_dir, "original_PCA_scores.npy"))
        )
        self.world_particles, self.local_particles  = self.load_particles(cfg.particles_dir)
        self.num_latent = int(np.load(os.path.join(cfg.pca_info_dir, "latent_dim.npy")))
        self.num_corr = self.local_particles[0].shape[0]
        self.num_pca = self.pca_scores[0].shape[0]
        np.savez(
            os.path.join(cfg.checkpoint_dir, "params.npz"),
            mean_score=self.mean_score,
            std_score=self.std_score,
            num_latent=self.num_latent,
            num_corr=self.num_corr,
            num_pca=self.num_pca
        )
            
    def __len__(self):
        return len(self.images)

    def load_particles(self, particle_dir):
        world_particle_list = []
        local_particle_list = []
        for file in os.listdir(particle_dir):
            if "local" in file:
                local_particle_list.append(os.path.join(particle_dir, file))
            if "world" in file:
                world_particle_list.append(os.path.join(particle_dir, file))
        
        world_particle_list = sorted(world_particle_list)
        local_particle_list = sorted(local_particle_list)

        world_particles = []
        local_particles = []

        for i in range(len(world_particle_list)):
            world_particles.append(get_particles(world_particle_list[i]))
            local_particles.append(get_particles(local_particle_list[i]))

        return [np.array(world_particles), np.array(local_particles)]

    def __getitem__(self, idx):
        image, _ = nrrd.read(self.images[idx])
        points_world = self.world_particles[idx] 
        points_local = self.local_particles[idx]
        image = torch.tensor(image).float().unsqueeze(0)
        ### only done for left artium example 
        image = F.pad(image, (8, 8, 0, 0, 0, 0), "constant", 0)
        #print(image.shape)
        return image, torch.tensor(points_world).float(), torch.tensor(points_local).float()
        