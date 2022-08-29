from cgi import test
import os
import yaml
import torch
import datetime

import sklearn.metrics as metrics
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from pathlib import Path

import torch.nn.functional as F
from utils.vis_utils import visualize_pred_gt, visualize_numpy, visualize_numpy_list
taxonomy_dic = {"02691156": "airplane", "02747177": "trash bin", "02773838": "bag", "02801938": "basket", "02808440": "bathtub", "02818832": "bed", 
                "02828884": "bench", "02843684": "birdhouse", "02871439": "bookshelf", "02876657": "bottle", "02880940": "bowl", "02924116": "bus", 
                "02933112": "cabinet", "02942699": "camera", "02946921": "can", "02954340": "cap", "02958343": "car", "02992529": "cellphone", "03001627": "chair", 
                "03046257": "clock", "03085013": "keyboard", "03207941": "dishwasher", "03211117": "display", "03261776": "earphone", "03325088": "faucet", 
                "03337140": "file cabinet", "03467517": "guitar", "03513137": "helmet", "03593526": "jar", "03624134": "knife", "03636649": "lamp", "03642806": "laptop", 
                "03691459": "loudspeaker", "03710193": "mailbox", "03759954": "microphone", "03761084": "microwaves", "03790512": "motorbike", "03797390": "mug", 
                "03928116": "piano", "03938244": "pillow", "03948459": "pistol", "03991062": "flowerpot", "04004475": "printer", "04074963": "remote", "04090263": "rifle", 
                "04099429": "rocket", "04225987": "skateboard", "04256520": "sofa", "04330267": "stove", "04379243": "table", "04401088": "telephone", "04460130": "tower", 
                "04468005": "train", "04530566": "watercraft", "04554684": "washer"}

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def check_speed(net, testloader):
    #### Measuring Inference Speed ####
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings=np.zeros((len(testloader),1))
    with torch.no_grad():
        for batch_idx, data_dic in enumerate(tqdm(testloader)):
            starter.record()
            data_dic = net(data_dic)
            ender.record()
           
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[batch_idx] = curr_time
    print('[INFERENCE TIMECOST] ', np.round(np.mean(timings), 5), ' ms')

def validate(net, testloader):
    net.eval()
    num_params = get_n_params(net)

    metric = Metrics('CDL2')

    fscore = []
    cdl1 = []
    cdl2 = []

    with torch.no_grad():
        for batch_idx, data_dic in enumerate(tqdm(testloader)):
            if len(data_dic['incomplete_pc'].shape)> 3:
                data_dic['incomplete_pc'] = data_dic['incomplete_pc'].squeeze(0)
            data_dic = net(data_dic)

            complete_pc_gt = data_dic['complete_pc']#.cpu().numpy()
            incomplete_pc_gt = data_dic['incomplete_pc']#.cpu().numpy()
            fine_pc_pred = data_dic['fine_pc_pred']#.cpu().numpy()

            complete_pc_pred = torch.cat((incomplete_pc_gt, fine_pc_pred), axis = 1)

            metrics = metric.get(complete_pc_pred, complete_pc_gt.repeat(complete_pc_pred.shape[0], 1, 1))

            fscore.append(metrics['fscore'])
            cdl1.append(metrics['cdl1'])
            cdl2.append(metrics['cdl2'])

        fscore = np.round(np.mean(fscore), 5)
        cdl1   = np.round(np.mean(cdl1), 5)
        cdl2   = np.round(np.mean(cdl2), 5)

        val_dic = {
            "fscore": fscore,
            "cdl1": cdl1,
            "cdl2": cdl2,
            "num_params": num_params,
        }

        return val_dic

import random
from pointnet2_ops import pointnet2_utils

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

def separate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
    '''
     separate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _,n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop,list):
            num_crop = random.randint(crop[0],crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:       
            center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).cuda()
        else:
            if isinstance(fixed_points,list):
                fixed_point = random.sample(fixed_points,1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1,1,3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

        idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

        crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop,list):
            INPUT.append(fps(input_data,2048))
            CROP.append(fps(crop_data,2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT,dim=0)# B N 3
    crop_data = torch.cat(CROP,dim=0)# B M 3

    return input_data.contiguous(), crop_data.contiguous()

from utils.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import open3d
# From PoinTr (Xumin Yu)
# Only measure CDL for fast evaluation

class Metrics(object):
    ITEMS = [{
        'name': 'CDL1',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel1',
        'eval_object': ChamferDistanceL1(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'CDL2',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel2',
        'eval_object': ChamferDistanceL2(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }]

    @classmethod
    def get(cls, pred, gt):
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            eval_func = eval(item['eval_func'])
            _values[i] = eval_func(pred, gt)

        metric_dic = {
            'fscore': 0.00,
            'cdl1': np.round(_values[0], 5),
            'cdl2': np.round(_values[1], 5)
        }

        return metric_dic

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        """pred and gt bs is 1"""
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)

        return ptcloud

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]

    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):

        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        b = pred.size(0)
        assert pred.size(0) == gt.size(0)
        if b != 1:
            f_score_list = []
            for idx in range(b):
                f_score_list.append(cls._get_f_score(pred[idx:idx+1], gt[idx:idx+1]))
            return sum(f_score_list)/len(f_score_list)
        else:
            pred = cls._get_open3d_ptcloud(pred)
            gt = cls._get_open3d_ptcloud(gt)

            dist1 = pred.compute_point_cloud_distance(gt)
            dist2 = gt.compute_point_cloud_distance(pred)

            recall = float(sum(d < th for d in dist2)) / float(len(dist2))
            precision = float(sum(d < th for d in dist1)) / float(len(dist1))
            return 2 * recall * precision / (recall + precision) if recall + precision else 0
    
    @classmethod
    def _get_chamfer_distancel1(cls, pred, gt):
        # chamfer_distance = cls.ITEMS[1]['eval_object']
        chamfer_distance = cls.ITEMS[0]['eval_object']
        return chamfer_distance(pred, gt).item() * 1000

    @classmethod
    def _get_chamfer_distancel2(cls, pred, gt):
        # chamfer_distance = cls.ITEMS[2]['eval_object']
        chamfer_distance = cls.ITEMS[1]['eval_object']

        return chamfer_distance(pred, gt).item() * 1000
    
    def __init__(self, metric_name):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value
