import os
import sys
import cv2
import numpy as np
from os.path import splitext, join, basename, abspath
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from utils.data_utils import load_img


def gen_img_dict():
    img_dir = "data/090303/test"
    gt_dir = 'data/090303/label-group1/'
    save_path = 'data/090303/img_cell_dict_test-1.npy'
    img_cell_dict = {}

    seqs = sorted(glob(join(img_dir, '*')))
    for i, seq in enumerate(seqs):
        seq_name = basename(seq)
        # print(seq_name, seq)
        gt_path = gt_dir + '090303_' + seq_name + '_all.txt' # name of your ground truth file(.txt)
        img_cell_dict = load_img(seq, gt_path, img_cell_dict, seq_name)

    print(img_cell_dict.keys())
    np.save(abspath(save_path), img_cell_dict)


if __name__ == '__main__':
    gen_img_dict()