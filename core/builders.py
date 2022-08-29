import os
import torch

from . import datasets, networks

def build_network(cfg):   
    net = networks.__all__[cfg.NETWORK.NAME](cfg)

    return net

def build_dataset(cfg, split = 'train', cls_choice = None):
    Dataset = datasets.__all__[cfg.DATASET.NAME.lower()](cfg, split = split, cls_choice = cls_choice)
    return Dataset

def build_optimizer(cfg, params, data_loader_length, mode = None):
    opt_cfg = cfg.OPTIMIZER
    
        
    if (opt_cfg.NAME.lower() == 'adam'):
        opt = torch.optim.Adam(params, lr=opt_cfg.LR, betas=opt_cfg.BETAS, weight_decay=opt_cfg.WEIGHT_DECAY)
    elif (opt_cfg.NAME.lower() == 'sgd'):
        opt = torch.optim.SGD(params, lr=opt_cfg.LR, momentum=opt_cfg.MOMENTUM, weight_decay=opt_cfg.WEIGHT_DECAY, nesterov=opt_cfg.NESTEROV)
    elif (opt_cfg.NAME.lower() == 'adamw'):
        opt = torch.optim.AdamW(params, lr=opt_cfg.LR, betas=opt_cfg.BETAS, weight_decay=opt_cfg.WEIGHT_DECAY)

    if (opt_cfg.SCHEDULER is not None):
        if opt_cfg.SCHEDULER.lower()== 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0 = opt_cfg.WARM_RESTART_EVERY * data_loader_length, eta_min = opt_cfg.MIN_LR)
    
    return opt, scheduler
        