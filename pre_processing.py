import logging
import os
from os.path import splitext, join, basename
from os import listdir
from glob import glob
import sys
import random
import numpy as np

import cv2
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

import torch.nn.functional as F

import hydra
from hydra.utils import to_absolute_path

import sys 
from data_loader import PP_Dataset
from net.unet import UNet_dvs
from utils.model_utils import RMSE_Q_NormLoss_img, eval_net_cla_img

def train_net(net, device, cfg):
    
    # loading img
    if cfg.dataset_params.eval.imgs is not None:
        train = PP_Dataset(cfg.dataset_params.data, cfg.dataset_params.train, 'train')
        val = PP_Dataset(cfg.dataset_params.data, cfg.dataset_params.eval, 'val')
        n_train = len(train)
        n_val = len(val)
    else:
        dataset = PP_Dataset(cfg.dataset_params.data, cfg.dataset_params.train, 'train')
        n_val = int(len(dataset) * cfg.dataset_params.eval.rate)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])

    epochs = cfg.train_params.epochs
    batch_size = cfg.train_params.batch_size
    lr = cfg.train_params.lr
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # optimizer & loss
    writer = SummaryWriter(log_dir=to_absolute_path('./logs'), comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    criterion = RMSE_Q_NormLoss_img(0.8)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(train)}
        Validation size: {len(val)}
        Checkpoints:     {cfg.output.save}
        Device:          {device.type}
    ''')
    
    net.train()
    # training 
    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['img']
                masks = batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.float32)

                mask_pred = net(imgs)
    
                loss = criterion(mask_pred, masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # loss.requires_grad = True
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(batch_size)
                global_step += 1
                if global_step % (n_train // (batch_size)) == 0 and epoch % 5 == 0:
                    # for tag, value in net.named_parameters():
                    #     tag = tag.replace('.', '/')
                    #     writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    #     writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_loss = eval_net_cla_img(net, val_loader, criterion, device, writer, epoch)
                    # scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info('Validation loss: {}'.format(val_loss))
                    writer.add_scalar('Loss/test', val_loss, global_step)

        if cfg.output.save:
            try:
                os.mkdir(to_absolute_path(cfg.output.dir))
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       to_absolute_path(os.path.join(cfg.output.dir, f'CP_epoch{epoch + 1}.pth')))
            torch.save(net,
                       to_absolute_path(os.path.join(cfg.output.dir, f'CP_model_epoch{epoch + 1}.pth')))
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

def pre_train(cfg):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet_dvs(n_channels=1, n_classes=1, bilinear=True)

    if cfg.load:
        net.load_state_dict(
            torch.load(cfg.load, map_location=device)
        )
        logging.info(f'Model loaded from {cfg.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  device=device,
                  cfg=cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

def load_model(model_path, parallel=False):
    # model_path = 
    # abs_model_path = abspath(model_path)
    # model = UNet_dvs(n_channels=1, n_classes=1, bilinear=True)
    # model.cuda()
    # state_dict = torch.load(model_path)
    # if parallel:
    #     model = torch.nn.DataParallel(model)
    # model.load_state_dict(state_dict, strict=False)
    model = torch.load(model_path)
    model.eval()
    return model

def pre_predict(cfg):
    MAXV = 255
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_files = sorted(glob(join(cfg.dataset_params.test.imgs, '*')))  
    # print(img_files)


    model_path = '/home/hzhang/zhanghaishan/docs/GraphQP/pp_model/CP_model_epoch100.pth'
    net = load_model(model_path)
    
    for i, i_path in enumerate(img_files):
        img = (cv2.imread(i_path, -1) / MAXV).astype('float32')
        img = np.expand_dims(img, axis=2).transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.cuda()
        output = net(img)
        mask_pred = output[0].cpu().detach().numpy()[0]
        # mask_pred = mask_pred.astype('int') * 255
        mask_pred = (mask_pred.max() - mask_pred)/(mask_pred.max() - mask_pred.min()) * 255
        mask_pred = mask_pred.astype('int')
        print(mask_pred.shape)
        # "/home/hzhang/zhanghaishan/docs/GraphQP/data/outputs_pp/"
        cv2.imwrite('/home/hzhang/zhanghaishan/docs/GraphQP/data/outputs_pp/'+str(i).zfill(4)+'.tif', mask_pred)
        # break


@hydra.main(config_path='config', config_name='pre_train')
def main(cfg):
    # pre_train(cfg)
    pre_predict(cfg)

    

if __name__ == '__main__':
    main()