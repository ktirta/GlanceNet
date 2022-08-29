from .shapenet55 import ShapeNet55
from .shapenet34 import ShapeNet34
from .shapenet21 import ShapeNet21
from .pcn import PCN
from .kitti import KITTI

__all__ = {
    'shapenet55': ShapeNet55,
    'shapenet34': ShapeNet34,
    'shapenet21': ShapeNet21,
    'pcn': PCN,
    'kitti': KITTI,
}