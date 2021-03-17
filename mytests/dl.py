import os, sys
sys.path.append("../")
import omegaconf
from torch_points3d.datasets.segmentation.s3dis import S3DIS1x1Dataset

cfg2 = omegaconf.OmegaConf.load("s3dis1x1.yaml")
dataset_cfg2 = cfg2.data
dataset_cfg2.dataroot = "/home/obarads/databox/torch_point3d/"

dataset2 = S3DIS1x1Dataset(dataset_cfg2)



