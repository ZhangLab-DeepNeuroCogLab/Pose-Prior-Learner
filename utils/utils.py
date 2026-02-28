import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np
import cv2
from typing import Union

def show_images(t, renorm):
    tensor = t
    if renorm:
        for i in range(tensor.shape[0]):
            #  bgr opencv to rgb
            tensor[i] = renorm(tensor[i])
    output_grid = make_grid(tensor, nrow=6, normalize=True, scale_each=True)
    return output_grid

def gen_grid2d(grid_size: int, left_end: float=-1, right_end: float=1) -> torch.Tensor:
    """
    Generate a grid of size (grid_size, grid_size, 2) with coordinate values in the range [left_end, right_end]
    """
    x = torch.linspace(left_end, right_end, grid_size)
    x, y = torch.meshgrid([x, x], indexing='ij')
    grid = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).reshape(grid_size, grid_size, 2)
    return grid

def draw_lines(paired_joints: torch.Tensor, heatmap_size: int=16, thick: Union[float, torch.Tensor]=1e-2) -> torch.Tensor:
    """
    Draw lines on a grid.
    :param paired_joints: (batch_size, n_points, 2, 2)
    :return: (batch_size, n_points, grid_size, grid_size)
    dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2
    """
    bs, n_points, _, _ = paired_joints.shape
    start = paired_joints[:, :, 0, :]   # (batch_size, n_points, 2)
    end = paired_joints[:, :, 1, :]     # (batch_size, n_points, 2)
    paired_diff = end - start           # (batch_size, n_points, 2)
    grid = gen_grid2d(heatmap_size).cuda().reshape(1, 1, -1, 2)
    diff_to_start = grid - start.unsqueeze(-2)  # (batch_size, n_points, heatmap_size**2, 2)
    # (batch_size, n_points, heatmap_size**2)
    t = (diff_to_start @ paired_diff.unsqueeze(-1)).squeeze(-1) / (1e-8+paired_diff.square().sum(dim=-1, keepdim=True))

    diff_to_end = grid - end.unsqueeze(-2)  # (batch_size, n_points, heatmap_size**2, 2)

    before_start = (t <= 0).float() * diff_to_start.square().sum(dim=-1)
    after_end = (t >= 1).float() * diff_to_end.square().sum(dim=-1)
    between_start_end = (0 < t).float() * (t < 1).float() * (grid - (start.unsqueeze(-2) + t.unsqueeze(-1) * paired_diff.unsqueeze(-2))).square().sum(dim=-1)

    squared_dist = (before_start + after_end + between_start_end).reshape(bs, n_points, heatmap_size, heatmap_size)
    heatmaps = torch.exp(- squared_dist / thick)
    return heatmaps
