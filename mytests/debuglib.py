import torch
from plyfile import PlyData, PlyElement
import numpy as np
from torch_geometric.datasets import S3DIS

def t2n(tensor: torch.tensor):
    """Convert torch.tensor to numpy.ndarray.

    Parameters
    ----------
    tensor : torch.tensor
        torch.tensor you want to convert to numpy.array.

    Returns
    -------
    array : numpy.ndarray
        numpy.ndarray converted
    """
    return tensor.cpu().detach().numpy()

def write_pc(filename, xyz, rgb=None):
    """
    write into a ply file
    ref.:https://github.com/loicland/superpoint_graph/blob/ssp%2Bspg/partition/provider.py
    """
    if rgb is None:
        # len(xyz[0]): for a xyz list, I don't use `.shape`.
        rgb = np.full((len(xyz), 3), 255, dtype=np.int32)
    if not isinstance(xyz, (np.ndarray, np.generic)):
        xyz = np.array(xyz, np.float32)

    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
