import torch
from torch import nn
import numpy as np
import nrrd
import math

def whiten_PCA_scores(scores):
	scores = np.array(scores)
	mean_score = np.mean(scores, 0)
	std_score = np.std(scores, 0)
	norm_scores = []
	for score in scores:
		norm_scores.append((score-mean_score)/std_score)
	return norm_scores, mean_score, std_score

def save_model():
	pass

def load_model():
	pass

def decompose_matrix(input_, device):
    M = input_.clone().transpose(0,1)
    scale = torch.zeros(3).to(device)
    angles = torch.zeros(3).to(device)
    
    translate = M[3, :3].clone()
    M[3, 3] = 0
    row = M[:3, :3].clone()
    scale[0] = torch.linalg.vector_norm(row[0]).item()
    row[0] /= scale[0]
    scale[1] = torch.linalg.vector_norm(row[1]).item()
    row[1] /= scale[1]
    scale[2] = torch.linalg.vector_norm(row[2]).item()
    row[2] /= scale[2]

    if torch.dot(row[0], torch.cross(row[1], row[2])) < 0:
        torch.negative(scale, scale)
        torch.negative(row, row)

    angles[1] = math.asin(-row[0, 2])
    if math.cos(angles[1]):
        angles[0] = math.atan2(row[1, 2], row[2, 2])
        angles[2] = math.atan2(row[0, 1], row[0, 0])
    else:
        angles[0] = math.atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0

    return scale, angles, translate

def euler_matrix(ai, aj, ak, device):

    Rx = torch.eye(3).to(device)
    Rx[1,1] = math.cos(ai)
    Rx[1,2] = -math.sin(ai)
    Rx[2,1] = math.sin(ai)
    Rx[2,2] = math.cos(ai)

    Ry = torch.eye(3).to(device)
    Ry[0,0] = math.cos(aj)
    Ry[0,2] = math.sin(aj)
    Ry[2,0] = -math.sin(aj)
    Ry[2,2] = math.cos(aj)

    Rz = torch.eye(3).to(device)
    Rz[0,0] = math.cos(ak)
    Rz[0,1] = -math.sin(ak)
    Rz[1,0] = math.sin(ak)
    Rz[1,1] = math.cos(ak)

    return Rz @ Ry @ Rx

def compose_matrix(scale=None, angles=None, translate=None, device="cpu"):
    M = torch.eye(4).to(device)
    if translate is not None:
        T = torch.eye(4).to(device)
        T[:3, 3] = translate[:3]
        M = M @ T
    if angles is not None:
        R = torch.eye(4).to(device)
        R[:3,:3] = euler_matrix(angles[0], angles[1], angles[2], device)
        M = M @ R
    if scale is not None:
        S = torch.eye(4).to(device)
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
        M = M @ S
    return M

    
    
def RMSEparticles(predicted, ground_truth):
    # numpy function for test evaluation 
    predicted = predicted.detach().cpu().squeeze().numpy()
    ground_truth = ground_truth.detach().cpu().squeeze().numpy()
    rmse = np.sqrt(np.mean((predicted - ground_truth)**2, 0))
    nrmse = rmse / np.sqrt(np.mean(ground_truth**2, 0))
    return np.mean(rmse)#, np.mean(nrmse)]

def read_dist_mat(vtk_file):
	with open(vtk_file) as f:
		lines = f.readlines()
	# find the index
	lines = np.array(lines)
	idx = 2+ np.where(lines == "FIELD FieldData 1\n")[0][0]
	dist = []
	for k in range(idx, len(lines)):
		spt = lines[k].split(' ')[:-1]
		for ss in spt:
			dist.append(float(ss))
	return dist
