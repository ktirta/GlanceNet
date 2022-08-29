import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import json
import open3d

# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py
class KITTI(data.Dataset):
    def __init__(self, cfg, split = 'test', cls_choice = None):
        self.cloud_path = 'data/KITTI/cars/%s.pcd'
        self.bbox_path = 'data/KITTI/bboxes/%s.txt'
        self.category_file = 'data/KITTI/KITTI.json'
        self.npoints = cfg.DATASET.NUM_POINTS
        self.subset = split
        assert self.subset == 'test'

        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
        self.transforms = Compose([{
                'callback': 'NormalizeObjectPose',
                'parameters': {
                    'input_keys': {
                        'ptcloud': 'partial_cloud',
                        'bbox': 'bounding_box'
                    }
                },
                'objects': ['partial_cloud', 'bounding_box']
            }, {
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'bounding_box']
            }])
       
        self.transforms_test = Compose([{
                'callback': 'NormalizeObjectPose',
                'parameters': {
                    'input_keys': {
                        'ptcloud': 'partial_cloud',
                        'bbox': 'bounding_box'
                    }
                },
                'objects': ['partial_cloud', 'bounding_box']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'bounding_box']
            }])
        self.file_list = self._get_file_list(self.subset)

    def _get_file_list(self, subset):
        """Prepare file list for the dataset"""
        file_list = []
        for dc in self.dataset_categories:
            samples = dc[subset]
            for s in samples:
                file_list.append({
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_cloud_path': self.cloud_path % s,
                    'bounding_box_path': self.bbox_path % s,
                })
        return file_list

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        for ri in ['partial_cloud', 'bounding_box']: 
            file_path = sample['%s_path' % ri]
            if ri == 'partial_cloud':
                pc = open3d.io.read_point_cloud(file_path)
                data[ri] = np.array(pc.points).astype(np.float32)
            else:
                data[ri] = np.loadtxt(file_path)
            # data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        data_dic = {
            'taxonomy_id': sample['taxonomy_id'],
            'model_id': sample['model_id'],
            'incomplete_pc': data['partial_cloud'].cuda(),
            'bbox': data['bounding_box']
        }

        return data_dic

import numpy as np
import torch
import transforms3d

class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = np.random.uniform(0, 1)
            if transform.__class__ in [NormalizeObjectPose]:
                data = transform(data)
            else:
                for k, v in data.items():
                    if k in objects and k in data:
                        if transform.__class__ in [
                            RandomMirrorPoints
                        ]:
                            data[k] = transform(v, rnd_value)
                        else:
                            data[k] = transform(v)

        return data

class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:    # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud


class RandomMirrorPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
        if rnd_value <= 0.25:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        elif rnd_value > 0.5 and rnd_value <= 0.75:
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class NormalizeObjectPose(object):
    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.ptcloud_key = input_keys['ptcloud']
        self.bbox_key = input_keys['bbox']

    def __call__(self, data):
        ptcloud = data[self.ptcloud_key]
        bbox = data[self.bbox_key]
        # Calculate center, rotation and scale
        # References:
        # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        data[self.bbox_key] = bbox
        return data
