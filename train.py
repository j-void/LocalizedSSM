import numpy as np
import torch
from util.data_loaders import *
import config as cfg
from networks.model_fullts import Image2Shape
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import random
import nrrd
from collections import defaultdict
import math
from torch.utils.tensorboard import SummaryWriter
import pickle



if __name__ == "__main__":
    
    writer = SummaryWriter(os.path.join(cfg.checkpoint_dir, "logs"))
    """
    Loads the training data into a dataloader
    """
    print("--------- Loading training data ---------")
    data = TrainDataset(cfg)
    train_size = int(0.9 * len(data))
    indices = list(range(len(data)))
    random.shuffle(indices)
    train_dataset = torch.utils.data.Subset(data, indices[:train_size])
    val_dataset = torch.utils.data.Subset(data, indices[train_size:])
    np.save(os.path.join(cfg.checkpoint_dir, "data_split.npy"), np.array(indices))
    """
    Creates a dataloader for the training data
    """
    train_loader = DataLoader(
			train_dataset,
			batch_size=cfg.batch_size,
			shuffle=True,
			num_workers=8,
            pin_memory=torch.cuda.is_available()
		)

    """
    Creates a dataloader for the validation data
    """
    val_loader = DataLoader(
			val_dataset,
			batch_size=cfg.batch_size,
			shuffle=False,
			num_workers=8,
            pin_memory=torch.cuda.is_available()
		)
    save_imdt = os.path.join(cfg.checkpoint_dir, "output")
    if not os.path.exists(save_imdt):
        os.makedirs(save_imdt)
    
    print("--------- Initializing Neworks ---------")
    model = Image2Shape(in_channels=1, out_channels=1, cfg=cfg, train_data=data, f_maps=16, num_levels=5)
    lambda_ = {"seg":100, "sim":0, "points_w":0, "points_l":0}
    min_loss = math.inf
    print("--------- Start Training ---------")
    train_w_loss = 0
    for epoch in tqdm(range(cfg.num_epochs), desc="Epochs: "):
        torch.cuda.empty_cache()
        model.lnet.train()
        model.dnet.train()
        loss_dict = defaultdict(list)
        loss_dict_train = defaultdict(list)
        
        if epoch == 10:
            for g in model.optimizer.param_groups:
                g['lr'] = 1e-5
                tqdm.write("Learning rate updated to - "+str(g['lr']))
            model.lnet.fc_loc[2].weight.data.zero_()
            model.lnet.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
            lambda_["sim"] = 100
            

        if epoch >= 20 and lambda_["points_w"] < 10:
            lambda_["points_w"] = (1.0e-9*(10**(epoch-20)))
        
        
        for idx, (img, seg, points_world, points_local, mat, mshape, ref) in enumerate((tqdm(train_loader, desc="Batch: "))):
            rnd_idx = random.sample(range(0, len(train_loader)), 5)
            model.optimizer.zero_grad()
            img = img.to(cfg.DEVICE)
            seg = seg.to(cfg.DEVICE)
            points_world = points_world.to(cfg.DEVICE)
            points_local = points_local.to(cfg.DEVICE)
            mat = mat.to(cfg.DEVICE)
            mshape = mshape.to(cfg.DEVICE)
            ref = ref.to(cfg.DEVICE)
            losses, outputs = model(img, seg, points_world, points_local, mshape, ref)
            loss_seg, loss_sim, loss_points_w = losses

            final_loss = loss_seg*lambda_["seg"]  + loss_sim*lambda_["sim"] + loss_points_w*lambda_["points_w"]

            final_loss.backward()
            model.optimizer.step()
            
            loss_dict["loss_seg"].append(loss_seg.detach().cpu()*lambda_["seg"])
            loss_dict["loss_sim"].append(loss_sim.detach().cpu()*lambda_["sim"])
            loss_dict["loss_points_w"].append(loss_points_w.detach().cpu()*lambda_["points_w"])
            loss_dict["final_loss"].append(final_loss.detach().cpu())
            
            loss_dict_train["loss_seg"].append(loss_seg.detach().cpu())
            loss_dict_train["loss_sim"].append(loss_sim.detach().cpu())
            loss_dict_train["loss_points_w"].append(loss_points_w.detach().cpu())
            
            
                
        loss_out = ""
        
        for key, val in loss_dict.items():
            mean_ = np.mean(np.array(val))
            loss_out += key + " = " + str(mean_) + ", "
            writer.add_scalar("Loss_train/"+key, mean_, epoch)
        tqdm.write("Epoch - "+str(epoch+1)+" : "+loss_out)
        
        # Validation every 5 epochs
        if epoch % 5 == 0:

            loss_dict_val = defaultdict(list)
            for idx, (img, seg, points_world, points_local, mat, mshape, ref) in enumerate(val_loader):
                img = img.to(cfg.DEVICE)
                seg = seg.to(cfg.DEVICE)
                points_world = points_world.to(cfg.DEVICE)
                points_local = points_local.to(cfg.DEVICE)
                mshape = mshape.to(cfg.DEVICE)
                ref = ref.to(cfg.DEVICE)
                outputs = model.val(img, seg, mshape, ref)
                loss_dict_val["loss_seg"].append(model.criterion_seg(outputs[0], seg).detach().cpu().squeeze().numpy())
                loss_dict_val["loss_sim"].append(model.criterion_sim(outputs[2], mshape).detach().cpu().squeeze().numpy())
                loss_dict_val["loss_world"].append(model.criterion_points(outputs[1], points_world).detach().cpu().squeeze().numpy())

            writer.add_scalars("Loss_val/segmentation", {'val':np.mean(np.array(loss_dict_val["loss_seg"])),
                                            'train': np.mean(np.array(loss_dict_train["loss_seg"]))}, epoch)
            writer.add_scalars("Loss_val/similarity", {'val':np.mean(np.array(loss_dict_val["loss_sim"])),
                                            'train': np.mean(np.array(loss_dict_train["loss_sim"]))}, epoch)                                
            writer.add_scalars("Loss_val/world", {'val':np.mean(np.array(loss_dict_val["loss_world"])),
                                            'train': np.mean(np.array(loss_dict_train["loss_points_w"]))}, epoch)
            
            if epoch >= 32:
                val_loss = np.mean(np.array(loss_dict_val["loss_world"]))
                if min_loss >= val_loss:
                    model.save_network("best")
                    tqdm.write("Current best found at epoch="+str(epoch+1))
                    min_loss = val_loss


        if epoch != 0 and epoch % 30 == 0:
            model.scheduler.step()
            

        

    print("--------- Done Training ---------")
    
    
    writer.flush()
    writer.close() 
    
                                
            
    
    
    
    

    