import os, sys

from torch_geometric import data
sys.path.append("../")
import omegaconf
from omegaconf import OmegaConf
from torch_points3d.datasets.segmentation.s3dis import S3DISOriginalFused, S3DISFusedDataset

tp3d_dataset_path = "/home/coder/workspace/data2/torch_point3d/"
data_config_path = '/home/coder/workspace/code/myrepo/torch-points3d/conf/data/segmentation/s3disfused-sparse.yaml'
config_path = '/home/coder/workspace/code/myrepo/torch-points3d/conf/config.yaml'
# dataset2 = S3DISOriginalFused(tp3d_dataset_path, test_area=5)
cfg = OmegaConf.load(config_path)
cfg.data = OmegaConf.load(data_config_path)
cfg.data.dataroot = tp3d_dataset_path
dataset = S3DISFusedDataset(cfg.data)

train_dataset = dataset.train_dataset
test_dataset = dataset.test_data
print(train_dataset)
print(train_dataset[0])

