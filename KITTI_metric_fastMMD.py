import os
import yaml

from easydict import EasyDict
import argparse
import torch
import numpy as np
import random

from utils.chamfer_dist import ChamferDistanceL2_split, ChamferDistanceL2, ChamferDistanceL1, ChamferDistanceL2_noreduce
from core.builders import build_dataset, build_network, build_optimizer
from utils.runtime_utils import cfg, cfg_from_yaml_file, validate, check_speed

from tqdm import tqdm

def build_ShapeNetCars():
    cfg_from_yaml_file('cfgs/pcn/PCN_Cars.yaml', cfg)

    train_dataset = build_dataset(cfg, split='train')
    val_dataset = build_dataset(cfg, split='val')
    test_dataset = build_dataset(cfg, split='test')
    
    CarsDataset = train_dataset + test_dataset + val_dataset

    return CarsDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_dir', 
        type = str, 
        help = 'path to kitti saved files dir')
    args = parser.parse_args()
    return args

def get_Fidelity():
    # Fidelity
    criterion = ChamferDistanceL2_split(ignore_zeros=True)

    metric = []
    for sample in Samples:
        input_data = torch.from_numpy(np.load(os.path.join(Data_path, sample, 'input.npy'))).unsqueeze(0).cuda()
        pred_data = torch.from_numpy(np.load(os.path.join(Data_path, sample, 'pred.npy'))).unsqueeze(0).cuda()
        metric.append(criterion(input_data, pred_data)[0])
    print('Fidelity is %f' % (sum(metric)/len(metric)))

def get_MMD():
    criterion = ChamferDistanceL2_noreduce(ignore_zeros=True)
    #MMD
    metric = []

    shapenet_len = len(ShapeNetCars_dataset)

    cache = []

    for gt_idx in tqdm(range(shapenet_len)):
        gt = ShapeNetCars_dataset[gt_idx]['complete_pc'].cuda()
        cache.append(gt)
    cache = torch.stack(cache)

    for item in tqdm(sorted(Samples)):
        pred_data = torch.from_numpy(np.load(os.path.join(Data_path, item, 'pred.npy'))).unsqueeze(0).cuda()
        min_cd = criterion(cache, pred_data.repeat(shapenet_len, 1, 1))
        min_cd = torch.min(min_cd)
        metric.append(min_cd)
        print('This item %s CD %f, MMD %f' % (item, min_cd, sum(metric)*1.0 / len(metric) ))
    print('MMD is %f' % (sum(metric)/len(metric)))

if __name__ == '__main__':

    random_seed = 0 # Setup seed for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    args = get_args()
    print('Loading Data...')
    ShapeNetCars_dataset = build_ShapeNetCars()
    CarsDataloader = torch.utils.data.DataLoader(ShapeNetCars_dataset, batch_size=24, shuffle = False, num_workers = 1)

    # Your data
    Data_path = args.exp_dir
    Samples = [item for item in os.listdir(Data_path) if os.path.isdir(Data_path + '/' + item)]
    criterion = ChamferDistanceL2_split(ignore_zeros=True)

    get_Fidelity()
    get_MMD()
