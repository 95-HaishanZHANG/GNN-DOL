import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from os.path import abspath

import torch.nn.functional as F
import numpy as np
import cv2

import hydra
from hydra.utils import to_absolute_path

from net.unet import UNet 


class RMSE_Q_NormLoss(nn.Module):
    def __init__(self, p_s):
        super().__init__()
        self.p_s = p_s

    def forward(self, yhat, y):
        x_pred = yhat.reshape(-1)
        x_gt = y.reshape(-1)
        bce_log = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.p_s]).cuda())
        
        return bce_log(x_pred, x_gt)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     # random.seed(seed)
     torch.backends.cudnn.deterministic = True

def load_model(model_path, parallel=False):
    model = UNet(1, 1)
    model.cuda()
    state_dict = torch.load(model_path)
    if parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def eval_net_cla(net, loader, criterion, device, writer, global_step):
    net.eval()
    n_val = len(loader)  # the number of batch
    total_error_n = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            g_data = batch.to(device=device)
            x_gt = g_data.matching_gt

            with torch.no_grad():
                gnn_pred = net(g_data)
                gnn_pred = gnn_pred.to(device=device)
            b_s = gnn_pred.shape[0]
            x_gt = x_gt.reshape(b_s, -1)

            error_n = criterion(gnn_pred, x_gt)
            total_error_n += error_n

            pbar.update()
            global_step += b_s

    net.train()
    return total_error_n / n_val

def compute_loss_cla(outputs, batch):
    output_n = outputs['classified_nodes'][0].view(-1)
    target_n = batch.node_labels.view(-1)
    output_n = torch.squeeze(output_n)
    loss_n = F.binary_cross_entropy_with_logits(output_n.cpu(), target_n.float().cpu())
    return loss_n

def save_output(file_path, output, global_step):
    if global_step < 0:
        np.savetxt(file_path, output, fmt='%.8f', delimiter=',', newline='\n', header=str(global_step))
    else:
        if output.is_cuda:
            output = output.cpu().detach().numpy()
        np.savetxt(file_path, output, fmt='%.8f', delimiter=',', newline='\n', header=str(global_step))
    return
