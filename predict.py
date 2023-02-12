import logging
import os
import sys
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
from utils.model_utils import RMSE_Q_NormLoss, setup_seed, save_output
from graph_construct import GMGraph
from torch_geometric.data import Data
from datetime import datetime
from net.GMNet import GMNet
import hydra
from hydra.utils import to_absolute_path as abs_path


def load_model(model_path, model_params, parallel=False):
    model = GMNet(model_params, 'test')
    model.cuda()
    state_dict = torch.load(model_path)
    if parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict, False)
    model.eval()
    return model

def load_fea(model, cell_info, img_path, frame_id=0):
    node_feats_f, node_ids = model.load_node_feats(cell_info, img_path, frame_id)
    node_neighbors_f = model.get_node_nghs(cell_info)
    edge_feats_f = model.get_edge_idx_P(cell_info, node_neighbors_f, frame_id)
    node_feats_f = torch.tensor(node_feats_f, dtype=torch.float)
    edge_feats_f = torch.tensor(edge_feats_f, dtype=torch.float)
    return node_feats_f, node_ids, edge_feats_f

def load_index(node_ids_cur, node_ids_nxt):
    idx_matching = []
    for k, id_cur in enumerate(node_ids_cur):
        for j, id_nxt in enumerate(node_ids_nxt):
            idx_matching.append([k, j])
    idx_matching = np.array(idx_matching)
    return torch.tensor(idx_matching, dtype=torch.long)

def predict(device, cfg):
    # loading data
    img_dict = abs_path(cfg.data.img_dict)
    img_dir = abs_path(cfg.data.imgs)
    out_save_path = abs_path(cfg.test.save_path)
    img_cell_dict = np.load(img_dict, allow_pickle=True).ravel()[0]
    f_inter = 1
    
    net = load_model(cfg.test.model_path, cfg.graph_model_params)
    g_graph = GMGraph(cfg.test.win_size, cfg.test.pp_model_path)

    for i, key_cur in enumerate(img_cell_dict.keys()):
        t_sta = datetime.now()
        id_key_cur = int(key_cur.split('-')[1])
        key_nxt = key_cur.split('-')[0] + '-' + str(id_key_cur + f_inter).zfill(4)
        if key_nxt not in img_cell_dict.keys():
            continue
        
        cell_info_cur = img_cell_dict[key_cur]
        img_path_cur = cell_info_cur[0]
        cell_info_nxt = img_cell_dict[key_nxt]
        img_path_nxt = cell_info_nxt[0]
        n_cur, n_id_cur, e_cur = load_fea(g_graph, cell_info_cur[2:], img_path_cur)
        n_nxt, n_id_nxt, e_nxt = load_fea(g_graph, cell_info_nxt[2:], img_path_nxt)
        idx_matching = load_index(n_id_cur, n_id_nxt)

        c_graph = Data(x1 = n_cur, x2 = n_nxt, 
                        e1 = e_cur[:,3:], e2 = e_nxt[:,3:], 
                        matching_idx = idx_matching,
                        num_x1 = len(n_id_cur), num_x2 = len(n_id_nxt))
        print('======================')
        with torch.no_grad():
            c_graph = c_graph.to(device)
            gnn_pred = net(c_graph)
            # You need to write your own output save path
            save_output(out_save_path, gnn_pred, -1)
                

@hydra.main(config_path='config', config_name='gm_test')
def main(cfg):

    torch.multiprocessing.set_start_method('spawn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predict(device=device, cfg=cfg)

if __name__ == '__main__':
    main()
