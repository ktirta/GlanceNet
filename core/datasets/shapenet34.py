import os
import torch
import numpy as np
import torch.utils.data as data

from utils.runtime_utils import separate_point_cloud
from utils.vis_utils import visualize_numpy

class ShapeNet34(data.Dataset):
    def __init__(self, cfg, split = 'train', cls_choice = None):
        self.cfg = cfg
        self.split = split
        self.npoints = cfg.DATASET.NUM_POINTS
        self.data_list_file = os.path.join('data/shapenet55-34/shapenet34/', f'{self.split}.txt')
        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        pc_data = np.load('data/shapenet55-34/shapenet_pc/' + sample['file_path']).astype(np.float32)
        pc_data = self.pc_norm(pc_data)
        pc_data = torch.from_numpy(pc_data).float().cuda()

        incomplete_pc, missing_pc = separate_point_cloud(pc_data.unsqueeze(0), self.cfg.DATASET.NUM_POINTS, self.cfg.DATASET.MIS_POINTS)
        incomplete_pc, missing_pc = incomplete_pc.squeeze(0), missing_pc.squeeze(0)

        data_dic = {
            'taxonomy_id': sample['taxonomy_id'],
            'model_id': sample['model_id'],
            'complete_pc': pc_data,
            'incomplete_pc': incomplete_pc,
            'missing_pc': missing_pc
        }
    
        return data_dic

    def __len__(self):
        return len(self.file_list)
