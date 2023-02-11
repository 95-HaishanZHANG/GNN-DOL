import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utils.model_utils import RMSE_Q_NormLoss, eval_net_cla, setup_seed, save_output
from net.GMNet import GMNet

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from data_loader import GM_Dataset
from torch.autograd import gradcheck


import hydra
from hydra.utils import to_absolute_path

def train_net(net, device, cfg, prob_size):
    # loading data
    if cfg.dataset_params.eval.csv is not None:
        cfg_data = cfg.dataset_params.train
        train = GM_Dataset(cfg_data, device, 'train', prob_size)
        cfg_data = cfg.dataset_params.eval
        val = GM_Dataset(cfg_data, device, 'test', prob_size)
        n_train = len(train)
        n_val = len(val)
    else:
        cfg_data = cfg.dataset_params.train
        dataset = GM_Dataset(cfg_data, device, 'train', prob_size)
        n_val = int(len(dataset) * cfg.dataset_params.eval.rate)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])

    epochs = cfg.train_params.epochs
    batch_size = cfg.train_params.batch_size
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    # optimizer & loss
    writer = SummaryWriter(log_dir=to_absolute_path('./logs'), comment=f'LR_{cfg.train_params.lr}_BS_{batch_size}')
    global_step = 0

    optimizer = optim.Adam(net.parameters(), lr=cfg.train_params.lr)
    criterion = RMSE_Q_NormLoss(prob_size)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Problem size:    {prob_size}
        Learning rate:   {cfg.train_params.lr}
        Training size:   {len(train)}
        Validation size: {len(val)}
        Checkpoints:     {cfg.output.save}
        Device:          {device.type}
    ''')

    # training 
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='graph') as pbar:
            for batch in train_loader:
                g_data = batch.to(device=device)
                x_gt = g_data.matching_gt
                outputs = net(g_data)
                outputs = outputs.to(device=device)
                loss = criterion(outputs, x_gt)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()}) 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(batch_size)
                global_step += 1
                if global_step % (n_train // (batch_size)) == 0:
                    val_loss = eval_net_cla(net, val_loader, criterion, device, writer, global_step)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info('Validation loss nodes: {}'.format(val_loss))
                    writer.add_scalar('Loss/test', val_loss, global_step)

        if cfg.output.save:
            try:
                os.mkdir(to_absolute_path(cfg.output.dir))
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       to_absolute_path(os.path.join(cfg.output.dir, f'CP_epoch{epoch + 1}.pth')))
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

@hydra.main(config_path='config', config_name='gm_train')
def main(cfg):
    torch.multiprocessing.set_start_method('spawn')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    setup_seed(20)

    prob_size = cfg.train_params.prob_size
    net = GMNet(cfg.graph_model_params, 'train', cfg.train_params.batch_size, prob_size)

    # if using a trained model
    if cfg.load:
        net.load_state_dict(
            torch.load(cfg.load, map_location=device)
        )
        logging.info(f'Model loaded from {cfg.load}')
    
    net.to(device=device)

    try:
        train_net(net=net,
                  device=device,
                  cfg=cfg,
                  prob_size=prob_size)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == '__main__':
    main()
