from debuglib import t2n, write_pc

import os, sys
from os.path import join as opj
sys.path.append("../")

import argparse
from omegaconf import OmegaConf
from torch_geometric import data
from torch_points3d.datasets.segmentation.s3dis import S3DIS1x1Ins
import k3d

# paths
tg_dataset_path = "/home/coder/workspace/data2/torch_point3d/s3dis1x1ins"
# data_config_path = '/home/coder/workspace/code/myrepo/torch-points3d/conf/data/segmentation/s3disfused.yaml'
# config_path = '/home/coder/workspace/code/myrepo/torch-points3d/conf/config.yaml'
output_path = 'output/'

# cfg = OmegaConf.load(config_path)
# cfg.data = OmegaConf.load(data_config_path)
# cfg.data.dataroot = tp3d_dataset_path

dataset = S3DIS1x1Ins(tg_dataset_path, test_area=5)

