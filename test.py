import torch
import argparse
import numpy as np
import random

from torch.utils.data import DataLoader

from core.builders import build_dataset, build_network
from utils.runtime_utils import cfg, cfg_from_yaml_file, validate, check_speed

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

args, cfg = parse_config()
exp_dir = ('/').join(args.ckpt.split('/')[:-2])

test_dataset = build_dataset(cfg, split='test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

# Build Network and Optimizer
net = build_network(cfg)
state_dict = torch.load(args.ckpt)
try:
    net.load_state_dict(state_dict['model_state_dict'])
except:
    net.load_state_dict(state_dict)

net = net.cuda()
net.eval()

random_seed = cfg.RANDOM_SEED # Setup seed for reproducibility
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


val_dict = validate(net, test_dataloader)
cdl1 = val_dict['cdl1']
cdl2 = val_dict['cdl2']
num_params = val_dict['num_params']

print('CDL1', cdl1)
print('CDL2', cdl2)
print('params', num_params)

# check_speed(net, test_dataloader) # to check the speed 

# with open(exp_dir + './eval_best.txt', 'w') as f:
#     f.write('\nBest CDL1: ' + str(cdl1))
#     f.write('\nBest CDL2: ' + str(cdl2))
#     f.write('\nparams' + str(num_params))