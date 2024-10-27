import cgi
import numpy as np
import torch
from util.data_loaders import *
import config as cfg
from networks.model_fullts import Image2Shape
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import nrrd
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from util.net_utils import RMSEparticles, read_dist_mat
import subprocess
import numpy as np
import shutil
import vtk
import shapeworks as sw

if __name__ == "__main__":
    print("--------- Loading data ---------")
    # Load the parameters from the training run
    train_params = np.load(os.path.join(cfg.checkpoint_dir, "params.npz"))
    # Load the data split
    indices = np.load(os.path.join(cfg.checkpoint_dir, "data_split.npy"))
    # Load the template mesh and particles
    template_mesh = cfg.template_mesh
    template_particles_w = cfg.template_particles_w
    template_particles_l = cfg.template_particles_l
    # Create a TrainDataset object
    data = TrainDataset(cfg)

    val_loader = DataLoader(
        data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=torch.cuda.is_available()
    )
    load_name = "best"

    print("--------- Initializing Neworks ---------")
    model = Image2Shape(in_channels=1, out_channels=1, cfg=cfg, train_data=data, f_maps=16, num_levels=5, fine_tune=True)

    # Load the pretrained model
    model.load_pretrained(load_name)
    model.eval()
    

    
    save_imdt = os.path.join(cfg.checkpoint_dir, "val_result", load_name, "val")
    if not os.path.exists(save_imdt):
        os.makedirs(save_imdt)
    wp_dir = os.path.join(save_imdt, "world", "particles")
    if not os.path.exists(wp_dir):
        os.makedirs(wp_dir)
    wm_dir = os.path.join(save_imdt, "world", "mesh")
    if not os.path.exists(wm_dir):
        os.makedirs(wm_dir)
    lp_dir = os.path.join(save_imdt, "local", "particles")
    if not os.path.exists(lp_dir):
        os.makedirs(lp_dir)
    lm_dir = os.path.join(save_imdt, "local", "mesh")
    if not os.path.exists(lm_dir):
        os.makedirs(lm_dir)
        
    val_errors = {"wp":[], "lp":[], "wm":[], "lm":[], "r":[], "s":[]}
    
    for idx, (img, seg, points_world, points_local, mat, mshape, ref) in enumerate((tqdm(val_loader, desc="Val: "))):
        img = img.to(cfg.DEVICE)
        seg = seg.to(cfg.DEVICE)
        points_world = points_world.to(cfg.DEVICE)
        points_local = points_local.to(cfg.DEVICE)
        mshape = mshape.to(cfg.DEVICE)
        ref = ref.to(cfg.DEVICE)
        # Get the predictions
        outputs = model.val(img, seg, mshape, ref)
        # Save the predicted world points
        np.savetxt(os.path.join(wp_dir, "real_"+str(idx)+".particles"), points_world.detach().cpu().squeeze().numpy())
        np.savetxt(os.path.join(wp_dir, "pred_"+str(idx)+".particles"), outputs[1].detach().cpu().squeeze().numpy())

        # Warp the template mesh to the predicted world points
        execCommand = ["shapeworks", 
			"warp-mesh", "--reference_mesh", template_mesh,
			"--reference_points", template_particles_w,
			"--target_points" ]
        execCommand.append(os.path.join(wp_dir, "real_"+str(idx)+".particles"))
        execCommand.append(os.path.join(wp_dir, "pred_"+str(idx)+".particles"))
        subprocess.check_call(execCommand)

        # Save the deformed mesh
        shutil.move(os.path.join(wp_dir, "real_"+str(idx)+".vtk"), os.path.join(wm_dir, "real_"+str(idx)+".vtk"))
        shutil.move(os.path.join(wp_dir, "pred_"+str(idx)+".vtk"), os.path.join(wm_dir, "pred_"+str(idx)+".vtk"))

        # Calculate the surface distance between the deformed mesh and the original mesh
        real_mesh = sw.Mesh(os.path.join(wm_dir, "real_"+str(idx)+".vtk"))
        pred_mesh = sw.Mesh(os.path.join(wm_dir, "pred_"+str(idx)+".vtk"))

        # Calculate the surface distance between the deformed mesh and the original mesh
        distance_values, cell_ids = real_mesh.distance(pred_mesh, method=sw.Mesh.DistanceMethod.PointToCell)
        # Save the distance values as a field in the predicted mesh
        pred_mesh.setField(name='distance', array=distance_values, type=sw.Mesh.FieldType.Point).write(os.path.join(wm_dir, "dist_"+str(idx)+".vtk"))
        # Calculate the average distance
        average_distance = np.mean(distance_values)
        
        # Store the errors
        val_errors["wm"].append(average_distance)
        # Calculate the error between the predicted world points and the ground truth world points
        val_errors["wp"].append(np.sqrt(model.criterion_points(outputs[1], points_world).detach().cpu().squeeze().numpy()))

        
        # Registration errors
        val_errors["r"].append(np.sqrt(model.criterion_sim(outputs[2], mshape).detach().cpu().squeeze().numpy()))
        
        # Segmentation errors
        val_errors["s"].append(np.sqrt(model.criterion_sim(outputs[0], seg).detach().cpu().squeeze().numpy()))

        filename_ = "out_iter_"+str(idx).zfill(5)
        nrrd.write(os.path.join(save_imdt, "pred_seg_"+filename_+".nrrd"), outputs[0].detach().cpu().squeeze().numpy())
        nrrd.write(os.path.join(save_imdt, "stn_seg_"+filename_+".nrrd"), outputs[2].detach().cpu().squeeze().numpy())

    # Get the best 3 and worst 3 errors on world particles
    val_errors_arr_w = np.array(val_errors["wp"])
    vidx_w = np.argsort(val_errors_arr_w)
    print("Best index for val world points=", vidx_w[:3], val_errors_arr_w[vidx_w[:3]])
    print("Worst index for val world points=", vidx_w[-3:], val_errors_arr_w[vidx_w[-3:]])
    val_errors_arr_wm = np.array(val_errors["wm"])
    vidx_wm = np.argsort(val_errors_arr_wm)
    print("Best index for val surface distance(world) =", vidx_wm[:3], val_errors_arr_wm[vidx_wm[:3]])
    print("Worst index for val surface distance(world) =", vidx_wm[-3:], val_errors_arr_wm[vidx_wm[-3:]])
    np.savez(os.path.join(save_imdt, "errors.npz"), w=val_errors_arr_w, wm=val_errors_arr_wm)

        
                

            

    print("--------- Done ---------")
                
                                
            
    
    
    
    

    