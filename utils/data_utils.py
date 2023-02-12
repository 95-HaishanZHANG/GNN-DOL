import os
import sys
import cv2
import numpy as np
from os.path import splitext, join, basename, abspath
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from utils.model_utils import load_model


def map_uint16_uint8(img):
    img = (img - img.min()) / (img.max() - img.min())
    img *= 255
    dst = img.astype(np.uint8)
    return dst 

def load_gt(gt_dir):
    ground_truth = np.genfromtxt(gt_dir, delimiter=' ')
    gt_list = [tuple(x) for x in ground_truth.tolist()]
    ground_truth = gt_list
    gt_dic = dict()
    for item in ground_truth:
        gt_dic.setdefault(item[0], []).append([item[1], item[2], item[3], item[4], item[5]])
    return gt_dic   

def load_img(img_dir, gt_dir, img_cell_dict, seq_name):
    '''
        save dict as the following format:
        {frame ID: [img url, cur_flag,
            [0. cell ID
            1. bbox position (x, y, x+w, y+w)
            2. cell position
            3. parent ID]
        ]}
    '''
    r = 25
    img_shape = [1042, 1390]
    gt_dic = load_gt(gt_dir)
    abs_img_dir = abspath(img_dir)
    img_path = sorted(glob(join(abs_img_dir, '*')))
    crop_img_posi = (r+1, r+1, (img_shape[0] - r-1), (img_shape[1] - r-1)) # 1042 1390
    for key_gt in gt_dic.keys():
        i = int(key_gt)
        if i > 200:
            continue
        sub_file_name = str(i).zfill(4) + '.tif'
        key_cell_dict = seq_name + '-' + str(i).zfill(4)
        for i_path in img_path:
            if sub_file_name in i_path:
                i_path_new = abs_img_dir + os.sep + sub_file_name
                os.rename(i_path, i_path_new)
                img_cell_dict.setdefault(key_cell_dict, []).append(i_path_new)
                break
        if key_cell_dict in img_cell_dict:
            gt_list = np.array(gt_dic[i])
            if np.sum(gt_list[:, 4]) < 0:
                img_cell_dict[key_cell_dict].append('cur')
            else:
                img_cell_dict[key_cell_dict].append('nxt')
            sort_idx = np.lexsort((gt_list[:, 2], gt_list[:, 1]))
            gt_list = gt_list[sort_idx, :]
            for gt_info in gt_list:
                cell_position = (int(gt_info[1]), int(gt_info[2]))
                if cell_position[1] > crop_img_posi[0] and cell_position[1] < crop_img_posi[2] and cell_position[0] > crop_img_posi[1] and cell_position[0] < crop_img_posi[3]:
                    bbox_position = (int(gt_info[1])-r, int(gt_info[2])-r, int(gt_info[1])+r, int(gt_info[2])+r)
                    img_cell_dict[key_cell_dict].append([int(gt_info[0]), bbox_position, cell_position, int(gt_info[3])])
            if len(img_cell_dict[key_cell_dict][2:]) < 10:
                print(img_cell_dict[key_cell_dict][0:2], img_cell_dict[key_cell_dict][2:], len(img_cell_dict[key_cell_dict][2:]), len(gt_list))
    
    return img_cell_dict

def load_raw_img(img_path, crop_img_bb):
    img = cv2.imread(img_path, -1)
    # print('img: ', img_path, img.shape)
    if img.dtype in ['uint16']:
        img = map_uint16_uint8(img)
    raw_img = img[crop_img_bb[0] : crop_img_bb[2], crop_img_bb[1] : crop_img_bb[3]]
    
    return raw_img

def load_embeddings_from_imgs(img_path, crop_img_bb, c_info, model_path):
    model = load_model(model_path)
    raw_img = load_raw_img(img_path, crop_img_bb)
    img_mask = np.zeros_like(raw_img)
    c_img = raw_img
    c_img = c_img.reshape(1, raw_img.shape[0], raw_img.shape[1]) 
    c_img = torch.from_numpy(c_img).to(torch.float32).unsqueeze(0)
    c_img = c_img.cuda()
    output_1, output_2 = model(c_img)
    c_embedding, c_output = output_1.cpu().detach().numpy(), output_2.cpu().detach().numpy()

    return raw_img, c_embedding, c_output.squeeze(0)



# if __name__ == '__main__':
#     # load_img()
#     # gen_img_dict()
#     # gen_img_dict_SA()
#     # gen_point_feature()
#     # gen_line_feature()
#     # generate_img_visual()
    