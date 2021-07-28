import os, sys
sys.path.append("../")
# from classes.dataset import S3DIS1x1wIns
import omegaconf

tp3d_dataset_path = "/home/obarads/databox/torch_point3d/"

cfg = omegaconf.OmegaConf.load("s3dis1x1.yaml")
dataset_cfg = cfg.data
dataset_cfg.dataroot = tp3d_dataset_path

from torch_points3d.datasets.segmentation.s3dis import S3DIS1x1Ins2, S3DISFusedDataset

dataset_mode_path = os.path.join(tp3d_dataset_path, "s3dis1x1ins2")

import torch
# raw_area_0 = torch.load(os.path.join(dataset_mode_path, "processed/raw_area_0.pt"))
# print(raw_area_0)
dataset = S3DIS1x1Ins2(dataset_mode_path, keep_instance=True)

# cfg = omegaconf.OmegaConf.load("s3disfused.yaml")
# dataset_cfg = cfg.data
# dataset_cfg.dataroot = tp3d_dataset_path
# dataset_cfg.sampling_format = "shpere"

# dataset = S3DISFusedDataset(dataset_cfg)


