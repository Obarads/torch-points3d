import os, sys
sys.path.append("../")

import argparse
from omegaconf import OmegaConf
from torch_geometric import data
from torch_points3d.datasets.segmentation.s3dis import S3DISOriginalFused, S3DISFusedDataset

def main():
    # paths
    tp3d_dataset_path = "/home/coder/workspace/data2/torch_point3d/"
    data_config_path = '/home/coder/workspace/code/myrepo/torch-points3d/conf/data/segmentation/s3disfused.yaml'
    config_path = '/home/coder/workspace/code/myrepo/torch-points3d/conf/config.yaml'

    parser = argparse.ArgumentParser('Test 01: use S3DISFusedDataset.')
    parser.add_argument('--dataset_path', '-d', type=str, default=tp3d_dataset_path)
    parser.add_argument('--data_config_path', '-dc', type=str, default=data_config_path)
    parser.add_argument('--config_path', '-c', type=str, default=config_path)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    cfg.data = OmegaConf.load(args.data_config_path)
    cfg.data.dataroot = args.dataset_path

    print(cfg.data)
    dataset = S3DISFusedDataset(cfg.data)

    train_dataset = dataset.train_dataset
    train_data = train_dataset[4]
    print(train_data.origin_id)

if __name__ == '__main__':
    main()
