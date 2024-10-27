import numpy as np
import os

DEVICE = "cuda"
img_dir = "datasets/left_atrium/images"
segmentation_dir = "datasets/left_atrium/segmentations"
properties_dir = "datasets/left_atrium/properties"
pca_info_dir = "datasets/left_atrium/ssm_output2/shape_models/PCA_Particle_Info"
particles_dir = "datasets/left_atrium/ssm_output2/shape_models/particles"
checkpoint_dir = "checkpoints/run29-8"
mean_shape = "datasets/left_atrium/reference.nrrd"
img_dims = [48, 160, 192]
crop_dims = [64, 64, 64]
batch_size = 1
num_epochs = 300
learning_rate = 1e-3

## for testing
template_particles_w = "datasets/left_atrium/reference_local.particles"
template_particles_l = "datasets/left_atrium/reference_local.particles"
template_mesh = "datasets/left_atrium/reference_mesh.vtk"
