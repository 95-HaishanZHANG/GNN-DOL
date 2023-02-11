import torch
import  torch.nn.functional as F

import numpy as np
import cv2
import hydra

from torch_scatter import scatter_min
from torch_geometric.data import Data
from concurrent.futures import ProcessPoolExecutor as Pool

from os.path import splitext, join, basename
from hydra.utils import to_absolute_path as abs_path
from skimage.measure import label, regionprops

from utils.data_utils import load_embeddings_from_imgs, load_raw_img


class GMGraph(object):
    
    def __init__(self, win_size, pp_model_path):
        self.w, self.h = win_size
        self.pp_model_path = pp_model_path

    def load_node_feats(self, cell_info, img_path, frame_id):
        # [num_nodes, num_node_feats]
        node_feats = []
        node_ids = []
        for i, c_info in enumerate(cell_info):
            # print('node_ids: ', c_info)
            node_ids.append([c_info[0], c_info[3]])
            crop_img_bb = ((c_info[2][1] - self.w), (c_info[2][0] - self.h), (c_info[2][1] + self.w), (c_info[2][0] + self.h)) # 1042 1390 (x, y, x+w, y+w)
            c_img, c_embedding, c_output = load_embeddings_from_imgs(img_path, crop_img_bb, c_info[1], self.pp_model_path)
            # c_img = c_img / 255.
            # c_img = (c_img - np.min(c_img)) / (np.max(c_img) - np.min(c_img))
            node_feats.append(c_output.flatten())
        node_feats = np.array(node_feats)
        # print('node_feats shape: ', node_feats.shape, len(node_ids))
        # print("======loading node finished!======")
        return node_feats, node_ids

    def get_edge_feats(self, cell_info):
        # calculate edge feats [cell position, distance of cell positon, of left upper, of right lower]
        c_edge_feat = []
        c_bbox_end = cell_info[1]
        c_pos_end = cell_info[2]

        # c_edge_feat.extend(c_pos_sta)
        c_edge_feat.extend(c_pos_end)
        # c_edge_feat.extend(c_bbox_sta)
        c_edge_feat.extend(c_bbox_end)
        # c_edge_feat.extend((c_bbox_sta[0] - c_pos_end[0], c_bbox_sta[1] - c_pos_end[1]))
        # c_edge_feat.extend((c_bbox_sta[2] - c_pos_end[0], c_bbox_sta[3] - c_pos_end[1]))
        return c_edge_feat
    
    def get_edge_idx_P(self, cell_info, node_neighbors, frame_id):
        # edge_idx = [] # [2, num_edges]
        edge_n_feats = []
        # edge_feats = [] # [num_edges, num_edge_features]
        for i, c_info_i in enumerate(cell_info):
            ngh_dst = np.array(node_neighbors[c_info_i[0]]).T
            # print(self.img_path[0], self.frame_id, ngh_dst.shape)
            ngh_idx = np.argsort(ngh_dst[1])[-2:]
            tmp_node_e_feat = [frame_id, c_info_i[0], c_info_i[3]]
            tmp_node_e_feat.extend(cell_info[i][2]) 
            tmp_node_e_feat.extend(cell_info[i][1]) 
            for j, c_info_j in enumerate(cell_info):
                if c_info_j[0] in ngh_dst[0][ngh_idx]:
                    # edge_idx.append([i, j])
                    c_edge_feat = self.get_edge_feats(c_info_j)
                    # print(c_edge_feat.shape)
                    # c_edge_feat = (c_edge_feat - np.min(c_edge_feat)) / (np.max(c_edge_feat) - np.min(c_edge_feat))
            #         edge_feats.append(c_edge_feat)
                    tmp_node_e_feat.extend(c_edge_feat)
            # tmp_node_e_feat = (tmp_node_e_feat - np.min(tmp_node_e_feat)) / (np.max(tmp_node_e_feat) - np.min(tmp_node_e_feat))
            edge_n_feats.append(tmp_node_e_feat)

        # edge_idx = np.array(edge_idx)
        # edge_feats = np.array(edge_feats)
        edge_n_feats = np.array(edge_n_feats)
        
        # print("======loading edge finished!======")
        # print('edge_feats: ', self.img_path[0], self.frame_id, edge_idx.shape, edge_feats.shape, edge_n_feats.shape)
        # print(edge_n_feats)
        return edge_n_feats
    
    def get_node_nghs(self, cell_info):
        ngh_ids = []
        # print('node_neighbors: ', node_neighbors.keys()) 
        node_neighbors = {} # {node id: [(ngh id, distance), ..., ()]}
        # print(self.img_path[0], self.frame_id, len(self.cell_info))
        for i in range(len(cell_info)):
            # print(self.img_path[0], self.frame_id, len(self.cell_info))
            for j in range(i+1, len(cell_info)):
                c_pos_sta = np.array(cell_info[i][2])
                c_pos_end = np.array(cell_info[j][2])
                c_dis = np.linalg.norm(c_pos_end - c_pos_sta)
                node_neighbors.setdefault(cell_info[i][0], []).append((cell_info[j][0], c_dis))
                node_neighbors.setdefault(cell_info[j][0], []).append((cell_info[i][0], c_dis)) 
        # print(self.img_path[0], self.frame_id, self.cell_info)
        return node_neighbors

    def get_matching_label(self, node_ids_cur, node_ids_nxt):
        x_gt = np.zeros((len(node_ids_cur), len(node_ids_nxt)))
        for k, id_cur in enumerate(node_ids_cur):
            for j, id_nxt in enumerate(node_ids_nxt):
                if id_cur[0] == id_nxt[0] or id_cur[0] == id_nxt[1]:
                    x_gt[k][j] = 1
        return torch.tensor(x_gt.T, dtype=torch.float).flatten()
    
    def construct_graph_object(self):
        """
        Constructs the entire Graph object to serve as input to the MPN, and stores it in self.graph_obj
        """
        # Load Appearance Data
        node_feats_cur, node_ids_cur = self._load_node_feats(self.cell_info_cur, self.img_path_cur)
        node_feats_nxt, node_ids_nxt = self._load_node_feats(self.cell_info_nxt, self.img_path_nxt)
        # node_ids.append((self.frame_id, -2))
        # print("shape of node_ids: ", len(node_ids))

        # Determine graph connectivity (i.e. edges) and compute edge features
        node_neighbors_cur = self._get_node_nghs(self.cell_info_cur)
        node_neighbors_nxt = self._get_node_nghs(self.cell_info_nxt)
        edge_n_feats_cur = self._get_edge_idx_P(self.cell_info_cur, node_neighbors_cur)
        edge_n_feats_nxt = self._get_edge_idx_P(self.cell_info_nxt, node_neighbors_nxt)
        x_gt = self._get_matching_label(node_ids_cur, node_ids_nxt)
        print("shape: ", node_feats_cur.shape, x_gt.shape)
       
        # Constrcut graph_obj
        self.graph_obj = Data(x1 = node_feats_cur, x2 = node_feats_nxt, 
                                e1 = edge_n_feats_cur, e2 = edge_n_feats_nxt,
                                matching_gt = x_gt)


@hydra.main(config_path='config', config_name='gm_construct')
def main(cfg):
    cfg_data = cfg.data
    cfg_type = cfg.params
    save_dir = abs_path(cfg_type.save_dir)
    imgs_dir = abs_path(cfg_data.imgs) # img_cell_dict
    img_dict = abs_path(cfg_data.img_dict)
    win_size = cfg_type.win_size
    pp_model_path = cfg_type.pp_model_path
    img_cell_dict = np.load(img_dict, allow_pickle=True).ravel()[0]
    g_graph = GMGraph(win_size, pp_model_path)

    pf_ids = []
    # print(imgs_dir)
    for i, key_dict in enumerate(img_cell_dict.keys()):
        if img_cell_dict[key_dict][1] in ['cur']:
            pf_ids.append(key_dict)
    # print(pf_ids)
    np.savetxt(save_dir+'/g_structure.txt', pf_ids, fmt='%s')

    c_x_list = {}
    # c_e_list = {}
    for i, key_dict in enumerate(img_cell_dict.keys()):
        # print(i, key_dict)
        cell_info = img_cell_dict[key_dict]
        img_path_cell = cell_info[0]
        save_key_c_list = save_dir +'/'+ key_dict.split('-')[0]
        frame_id = int(key_dict.split('-')[1])
        # print(key_dict, frame_id, save_key_c_list)
        if len(cell_info) <= 2:
            continue    
        cell_info_f = cell_info[2:]
        node_feats_f, _ = g_graph.load_node_feats(cell_info_f, img_path_cell, frame_id)
        node_neighbors_f = g_graph.get_node_nghs(cell_info_f)
        edge_feats_f = g_graph.get_edge_idx_P(cell_info_f, node_neighbors_f, frame_id)
        edge_feats_f = np.hstack((edge_feats_f, node_feats_f))
        print('shape: ', edge_feats_f.shape, node_feats_f.shape)
        c_x_list.setdefault(save_key_c_list, []).extend(edge_feats_f)
        # c_e_list.setdefault(save_key_c_list, []).append(edge_feats_f)
    # print('c_x_list: ', c_x_list.values())

    for key_x in c_x_list.keys():
        c_info = np.array(c_x_list[key_x])
        # print(c_info.shape)
        np.savetxt(key_x+'_xe_fea.csv', c_info, delimiter=',')
    print('======x finish========')



if __name__ == '__main__':
    main()