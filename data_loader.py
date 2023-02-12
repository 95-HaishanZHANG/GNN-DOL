import enum
from os.path import splitext, join, basename, exists
from os import listdir
import numpy as np
from glob import glob
import torch
import scipy.io as scio
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
import logging
import cv2
import random
from scipy.ndimage import rotate
from hydra.utils import to_absolute_path as abs_path
from graph_construct import GMGraph
from utils.data_utils import load_embeddings_from_imgs
from scipy.ndimage.filters import gaussian_filter
        

class GM_Dataset(InMemoryDataset):
    def __init__(self, cfg_data, device, mode, prob_size):
        assert mode in ('train', 'val', 'test')
        self.g_stc = abs_path(cfg_data.g_stc) # txt file
        self.fea_csv = abs_path(cfg_data.csv)
        self.prob_size = prob_size
        self.mode = mode
        self.device = device

        g_txt = np.genfromtxt(self.g_stc, dtype='str')
        self.ids = {}
        for item in g_txt:
            seq_name, f_id = item.split('-')
            self.ids.setdefault(seq_name, []).append([seq_name+'-'+str(int(f_id)), 
                                                        seq_name+'-'+str(int(f_id)+1)])

        self.x_fea_dict = {}
        for key_dict in self.ids.keys():
            csv_path_x = self.fea_csv + '/' + str(key_dict) + '_xe_fea.csv'
            if not exists(csv_path_x):
                continue
            csv_file_x = np.genfromtxt(csv_path_x, delimiter=',')
            for item in csv_file_x:
                x_key = key_dict + '-' + str(int(item[0]))
                self.x_fea_dict.setdefault(x_key, []).append(item[1:])
        
        super(GM_Dataset, self).__init__(self.fea_csv) 
        read_path = self.fea_csv + '/' + self.processed_paths[0].split('/')[-1]
        self.data, self.slices = torch.load(read_path)


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        """
        produce file name when taking into account the type of the processed graph
        """
        pt_path = self.fea_csv + '/' + 'gm_data_test.pt'
        return [pt_path]

    def download(self):
        pass

    def get_matching_label(self, node_ids_cur, node_ids_nxt):
        x_gt = np.zeros((len(node_ids_cur), len(node_ids_nxt)))
        idx_gt = []
        for k, id_cur in enumerate(node_ids_cur):
            for j, id_nxt in enumerate(node_ids_nxt):
                idx_gt.append([k, j])
                if id_cur[0] == id_nxt[0] or id_cur[0] == id_nxt[1]:
                    x_gt[k][j] = 1
        idx_gt = np.array(idx_gt)
        return torch.tensor(x_gt.T, dtype=torch.float).flatten(), torch.tensor(idx_gt, dtype=torch.long)
    
    def align_data(self, x_cur, x_nxt):
        x_cur = np.array(x_cur)
        x_nxt = np.array(x_nxt)
        par_id_nxt = x_nxt[:, 1]
        cell_id_cur = x_cur[:, 0]
        exist_dvs_id = np.intersect1d(cell_id_cur, par_id_nxt)
        
        idx_cell_nxt = np.array([], dtype=int)
        if exist_dvs_id.shape[0] != 0:
            idx_cell_nxt = np.where(np.isin(par_id_nxt, exist_dvs_id))[0] # ID of selected daughter cells
            idx_cell_nxt = idx_cell_nxt[:self.prob_size] if idx_cell_nxt.shape[0] > self.prob_size else idx_cell_nxt
        num_no_dvs = (self.prob_size - idx_cell_nxt.shape[0]) if (self.prob_size - idx_cell_nxt.shape[0]) > 0 else 0
        no_par_idx = np.delete(range(len(par_id_nxt)), idx_cell_nxt)
        if num_no_dvs != 0:
            idx_no_dvs_nxt = np.random.choice(no_par_idx, size=num_no_dvs, replace=False)
            idx_cell_nxt = np.append(idx_cell_nxt, idx_no_dvs_nxt)
        x_nxt_new = x_nxt[idx_cell_nxt]
        
        cell_id_sel = x_nxt_new[:, 0]
        par_id_sel = x_nxt_new[:, 1]
        idx_cell_cur = np.where(np.isin(cell_id_cur, cell_id_sel))[0]
        idx_par_cur = np.where(np.isin(cell_id_cur, par_id_sel))[0]
        if idx_par_cur.shape[0] != 0:
            idx_cell_cur = np.append(idx_cell_cur, idx_par_cur)
        num_cur = idx_cell_cur.shape[0]
        num_cur_sup = (self.prob_size - num_cur) if (self.prob_size - num_cur) > 0 else 0
        if num_cur_sup != 0:
            idx_cur_sup = np.random.choice(np.delete(range(len(cell_id_cur)), idx_cell_cur), size=num_cur_sup, replace=False)
            idx_cell_cur = np.append(idx_cell_cur, idx_cur_sup)
        x_cur_new = x_cur[idx_cell_cur]
        x_nxt_new = x_nxt[idx_cell_nxt]
        return x_cur_new, x_nxt_new

    def get_xe_fea(self, x_cur, x_nxt):
        node_ids_cur = []
        node_ids_nxt = []
        x_fea_cur = []
        e_fea_cur = []
        x_fea_nxt = []
        e_fea_nxt = []
        for item in x_cur:
            node_ids_cur.append([item[0], item[1]])
            e_fea_cur.append(item[2:20])
            x_fea_cur.append(item[20:])
        for item in x_nxt:
            node_ids_nxt.append([item[0], item[1]])
            e_fea_nxt.append(item[2:20])
            x_fea_nxt.append(item[20:])
        x_fea_cur, x_fea_nxt, e_fea_cur, e_fea_nxt, node_ids_cur, node_ids_nxt = map(np.array, [x_fea_cur, x_fea_nxt, e_fea_cur, e_fea_nxt, node_ids_cur, node_ids_nxt])
        x_fea_cur = torch.tensor(x_fea_cur, dtype=torch.float)
        x_fea_nxt = torch.tensor(x_fea_nxt, dtype=torch.float)
        e_fea_cur = torch.tensor(e_fea_cur, dtype=torch.float)
        e_fea_nxt = torch.tensor(e_fea_nxt, dtype=torch.float)
        return x_fea_cur, x_fea_nxt, e_fea_cur, e_fea_nxt, node_ids_cur, node_ids_nxt

    def process(self):
        data_list = []
        for key_info in self.ids.keys():
            g_info = self.ids[key_info]
            for item in g_info:
                key_cur = item[0]
                key_nxt = item[1]
                if key_cur not in self.x_fea_dict.keys():
                    continue
                x_cur = self.x_fea_dict[key_cur]
                x_nxt = self.x_fea_dict[key_nxt]
                x_cur, x_nxt = self.align_data(x_cur, x_nxt)
                x_fea_cur, x_fea_nxt, e_fea_cur, e_fea_nxt, node_ids_cur, node_ids_nxt = self.get_xe_fea(x_cur, x_nxt)
                x_gt, idx_gt = self.get_matching_label(node_ids_cur, node_ids_nxt)
                c_graph = Data(x1 = x_fea_cur, x2 = x_fea_nxt, 
                                    e1 = e_fea_cur, e2 = e_fea_nxt, 
                                    matching_gt = x_gt, matching_idx = idx_gt,
                                    num_x1 = len(node_ids_cur), num_x2 = len(node_ids_nxt))
                data_list.append(c_graph)
        print(f"Num of produced graphs is {len(data_list)}")        
        file_name = self.processed_paths[0].split('/')[-1]
        write_path = self.fea_csv +'/'+ file_name
        torch.save(self.collate(data_list), write_path)
       
    


if __name__ == '__main__':
    gm = GM_Dataset()
    