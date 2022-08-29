import os
import torch
import argparse
import random
import shutil
import numpy as np

from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from core.builders import build_dataset, build_network

from utils.runtime_utils import cfg, cfg_from_yaml_file

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--save_dir', type=str, default=None, help='path to dir')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def save_preds(net, testloader, save_dir):
    net.eval()

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, data_dic in enumerate(tqdm(testloader)):
            data_dic = net(data_dic)
            
            incomplete_pc_gt = data_dic['incomplete_pc']#.cpu().numpy()
            fine_pc_pred = data_dic['fine_pc_pred']#.cpu().numpy()

            sample_name = data_dic['model_id'][0]
            sample_dir = os.path.join(save_dir, sample_name)
            os.makedirs(sample_dir, exist_ok=True)

            # Scaling Back
            min_pred_x, max_pred_x = torch.min(fine_pc_pred[0, :, 0]), torch.max(fine_pc_pred[0, :, 0])
            min_pred_y, max_pred_y = torch.min(fine_pc_pred[0, :, 1]), torch.max(fine_pc_pred[0, :, 1])
            min_pred_z, max_pred_z = torch.min(fine_pc_pred[0, :, 2]), torch.max(fine_pc_pred[0, :, 2])
            pred_xl, pred_yl, pred_zl = max_pred_x - min_pred_x, max_pred_y - min_pred_y, max_pred_z - min_pred_z
            
            bbox = data_dic['bbox']
            min_gt_x, max_gt_x = torch.min(bbox[0, :, 0]), torch.max(bbox[0, :, 0])
            min_gt_y, max_gt_y = torch.min(bbox[0, :, 1]), torch.max(bbox[0, :, 1])
            min_gt_z, max_gt_z = torch.min(bbox[0, :, 2]), torch.max(bbox[0, :, 2])               
            gt_xl, gt_yl, gt_zl = max_gt_x - min_gt_x, max_gt_y - min_gt_y, max_gt_z - min_gt_z
            
            dx, dy, dz = pred_xl / gt_xl, pred_yl / gt_yl, pred_zl / gt_zl
            
            incomplete_pc_gt[:, :, 0] = incomplete_pc_gt[:, :, 0]*dx 
            incomplete_pc_gt[:, :, 1] = incomplete_pc_gt[:, :, 1]*dy
            incomplete_pc_gt[:, :, 2] = incomplete_pc_gt[:, :, 2]*dz

            complete_pc_pred = torch.cat((incomplete_pc_gt, fine_pc_pred), axis = 1)

            np.save(os.path.join(sample_dir, 'input.npy'), incomplete_pc_gt.squeeze(0).cpu().numpy())
            np.save(os.path.join(sample_dir, 'pred.npy'), complete_pc_pred.squeeze(0).cpu().numpy())

args, cfg = parse_config()
exp_dir = ('/').join(args.ckpt.split('/')[:-2])

random_seed = cfg.RANDOM_SEED # Setup seed for reproducibility
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Build Dataloader
val_dataset = build_dataset(cfg, split='test')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

# Build Network and Optimizer
net = build_network(cfg)
state_dict = torch.load(args.ckpt)
try:
    net.load_state_dict(state_dict['model_state_dict'])
except:
    net.load_state_dict(state_dict)
net = net.cuda()
net.eval()

save_dir = args.save_dir
val_dict = save_preds(net, val_dataloader, save_dir=save_dir,)

