import os
import torch
import argparse
import random
import shutil
import numpy as np

from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from core.builders import build_dataset, build_network, build_optimizer

from utils.runtime_utils import cfg, cfg_from_yaml_file, validate, check_speed
from utils.vis_utils import visualize_numpy

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--exp_name', type=str, default=None, help='specify experiment name for saving outputs')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed number')
    parser.add_argument('--val_steps', type=int, default=1, help='perform validation every n steps')
    parser.add_argument('--pretrained_ckpt', type = str, default = None, help='path to pretrained ckpt')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    exp_dir = cfg.ROOT_DIR / 'experiments' / cfg.DATASET.NAME / args.exp_name
    os.makedirs(exp_dir, exist_ok=True)
    shutil.copy2(args.cfg_file, exp_dir)

    return args, cfg

args, cfg = parse_config()

random_seed = cfg.RANDOM_SEED # Setup seed for reproducibility
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Build Dataloader
train_dataset = build_dataset(cfg, split = 'train')
train_dataloader = DataLoader(train_dataset, batch_size=cfg.OPTIMIZER.BATCH_SIZE, shuffle=True, drop_last=True)

val_dataset = build_dataset(cfg, split='test')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

if cfg.DATASET.NAME.lower() == 'shapenet34':
    cfg.DATASET.NAME = 'ShapeNet21'
    val21_dataset = build_dataset(cfg, split = 'test')
    val21_dataloader = DataLoader(val21_dataset, batch_size=1, shuffle=False, drop_last=False)
    cfg.DATASET.NAME = 'ShapeNet34'


net = build_network(cfg).cuda()
opt, scheduler = build_optimizer(cfg, net.parameters(), len(train_dataloader))

from torch.utils.tensorboard import SummaryWriter
ckpt_dir = cfg.ROOT_DIR / 'experiments' / cfg.DATASET.NAME / args.exp_name / 'ckpt'
tensorboard_dir = cfg.ROOT_DIR / 'experiments' / cfg.DATASET.NAME / args.exp_name / 'tensorboard'

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

writer = SummaryWriter(tensorboard_dir)

min_loss = 1e20
min_cdl1 = 1e20
min_cdl2 = 1e20
max_fscore = 0

min21_loss = 1e20
min21_cdl1 = 1e20
min21_cdl2 = 1e20
max21_fscore = 0

steps_cnt = 0
epoch_cnt = 0

# check_speed(net, val_dataloader)
# validate(net, val_dataloader, None, None)
for epoch in tqdm(range(1, cfg.OPTIMIZER.MAX_EPOCH + 1)):
    opt.zero_grad()
    net.zero_grad()
    net.train()
    loss = 0
    
    for data_dic in tqdm(train_dataloader):
        data_dic = net(data_dic)

        loss, loss_dic = net.get_loss(data_dic)
        loss.backward()

        steps_cnt += 1

        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
        opt.step()
        opt.zero_grad()
        net.zero_grad()
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        total_loss = loss_dic['loss']
        fine_loss = loss_dic['fine_loss']
        coarse_loss = loss_dic['coarse_loss']
        mid_loss = loss_dic['mid_loss']
  
        writer.add_scalar('steps/loss', total_loss, steps_cnt)
        writer.add_scalar('steps/coarse_loss', coarse_loss, steps_cnt)
        writer.add_scalar('steps/fine_loss', fine_loss, steps_cnt)
        writer.add_scalar('steps/mid_loss', mid_loss, steps_cnt)
        writer.add_scalar('steps/lr', lr, steps_cnt)

    if ((epoch % args.val_steps) == 0) & (epoch > cfg.OPTIMIZER.START_EVAL_AFTER_EPOCH):
        val_dic = validate(net, val_dataloader)

        print('='*20, 'Epoch ' + str(epoch), '='*20)

        writer.add_scalar('epochs/val_fscore', val_dic['fscore'], epoch_cnt)
        writer.add_scalar('epochs/val_cdl1', val_dic['cdl1'], epoch_cnt)
        writer.add_scalar('epochs/val_cdl2', val_dic['cdl2'], epoch_cnt)

        print(  'Val F-Score: ', val_dic['fscore'], 
                'Val CDL1: ', val_dic['cdl1'], 
                'Val CDL2: ', val_dic['cdl2'],
                'Timecost: ', val_dic['time_ms'])

        epoch_cnt += 1

        if val_dic['fscore'] > max_fscore:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'cdl1': val_dic['cdl1'],
                'cdl2': val_dic['cdl2'],
                'fscore': val_dic['fscore']
                }, ckpt_dir / 'ckpt-best-fscore.pth')
            
            max_f1 = val_dic['fscore']

        if val_dic['cdl1'] < min_cdl1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'cdl1': val_dic['cdl1'],
                'cdl2': val_dic['cdl2'],
                'fscore': val_dic['fscore']
                }, ckpt_dir / 'ckpt-best-cdl1.pth')
            min_cdl1 = val_dic['cdl1']

        if val_dic['cdl2'] < min_cdl2:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'cdl1': val_dic['cdl1'],
                'cdl2': val_dic['cdl2'],
                'fscore': val_dic['fscore']
                }, ckpt_dir / 'ckpt-best-cdl2.pth')     
            min_cdl2 = val_dic['cdl2']

        torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'cdl1': val_dic['cdl1'],
                    'cdl2': val_dic['cdl2'],
                    'fscore': val_dic['fscore']
                    }, ckpt_dir / 'ckpt-last.pth')

        if cfg.DATASET.NAME.lower() == 'pcn':
            save_name = 'ckpt-ep' + str(epoch) + '-cd' + str(np.round(val_dic['cdl2'], 2).item())
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'cdl1': val_dic['cdl1'],
                    'cdl2': val_dic['cdl2'],
                    'fscore': val_dic['fscore']
                    }, ckpt_dir / save_name)

        if cfg.DATASET.NAME.lower() == 'shapenet34':
            val_21_dic = validate(net, val21_dataloader)
            writer.add_scalar('epochs/shapenet21/val_fscore', val_21_dic['fscore'], epoch_cnt)
            writer.add_scalar('epochs/shapenet21/val_cdl1', val_21_dic['cdl1'], epoch_cnt)
            writer.add_scalar('epochs/shapenet21/val_cdl2', val_21_dic['cdl2'], epoch_cnt)

            print(  '\nVal21 F-Score: ', val_21_dic['fscore'], 
                    'Val21 CDL1: ', val_21_dic['cdl1'], 
                    'Val21 CDL2: ', val_21_dic['cdl2'],
                )

            epoch_cnt += 1

            if val_21_dic['fscore'] > max21_fscore:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'cdl1': val_21_dic['cdl1'],
                    'cdl2': val_21_dic['cdl2'],
                    'fscore': val_21_dic['fscore']
                    }, ckpt_dir / 'ckpt21-best-fscore.pth')
                
                max21_f1 = val_21_dic['fscore']

            if val_21_dic['cdl1'] < min21_cdl1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'cdl1': val_21_dic['cdl1'],
                    'cdl2': val_21_dic['cdl2'],
                    'fscore': val_21_dic['fscore']
                    }, ckpt_dir / 'ckpt21-best-cdl1.pth')
                min21_cdl1 = val_21_dic['cdl1']

            if val_21_dic['cdl2'] < min21_cdl2:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'cdl1': val_21_dic['cdl1'],
                    'cdl2': val_21_dic['cdl2'],
                    'fscore': val_21_dic['fscore']
                    }, ckpt_dir / 'ckpt21-best-cdl2.pth')     
                min21_cdl2 = val_21_dic['cdl2']