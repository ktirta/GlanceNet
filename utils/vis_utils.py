import numpy as np
import open3d as o3d

import torch
from tqdm import tqdm
import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d


def visualize_pred_gt(pred, gt):
    x_pred = pred[0, :, 0]
    y_pred = pred[0, :, 1]
    z_pred = pred[0, :, 2]

    x_gt = gt[0, :, 0]
    y_gt = gt[0, :, 1]
    z_gt = gt[0, :, 2]

    fig = plt.figure(figsize=(8, 8))
    ax_pred = fig.add_subplot(121, projection='3d')
    ax_gt = fig.add_subplot(122, projection='3d')

    ax_pred.scatter(x_pred, y_pred, z_pred)
    ax_pred.axis('off')
    ax_pred.grid(b=None)

    ax_gt.scatter(x_gt, y_gt, z_gt)
    ax_gt.axis('off')
    ax_gt.grid(b=None)
    
    plt.show()

def visualize_numpy(pc_numpy, colors = None, save_path = None):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc_numpy)
    try:
        point_cloud.colors = o3d.utility.Vector3dVector(np.full((pc_numpy.shape[0], 3), colors))
    except:
        pass

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    ctr = vis.get_view_control()
    ctr.set_up((1, 0, 0))
    ctr.set_front((0, 1, 0))

    vis.run()

    if save_path is not None:
        # vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)

def visualize_numpy_with_boxes(pc_numpy, corners_list):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc_numpy)
    try:
        point_cloud.colors = o3d.utility.Vector3dVector(np.full((pc_numpy.shape[0], 3), colors))
    except:
        pass

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)

    line_set_list = []

    for corners in corners_list:
        lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4,5], [5, 6], [6,7], [7,4], [4,3], [7,2], [5,0], [6,1], [8,9]]
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set_list.append(line_set)

        vis.add_geometry(line_set)

    ctr = vis.get_view_control()
    ctr.set_up((1, 0, 0))
    ctr.set_front((0, 1, 0))

    vis.run()

def visualize_numpy_list(pc_numpy_list, color_list = None, save_path=None):
    assert(len(pc_numpy_list) == len(color_list))

    point_cloud = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    final_pc = []
    final_colors = []
    for pc_numpy, colors in zip(pc_numpy_list, color_list):
        colors = np.full((len(pc_numpy), 3), colors)
        final_pc.append(pc_numpy)
        final_colors.append(colors)
    
    final_pc = np.concatenate(final_pc)
    final_colors = np.concatenate(final_colors)

    point_cloud.points = o3d.utility.Vector3dVector(final_pc)
    point_cloud.colors = o3d.utility.Vector3dVector(final_colors)
    vis.add_geometry(point_cloud)


    ctr = vis.get_view_control()
    ctr.set_up((0, 1, 0))
    ctr.set_front((1, 0, 0))
    ctr.camera_local_rotate(0.2, 0.2)
    if save_path is not None:
        # vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)

    vis.run()
    vis.destroy_window()