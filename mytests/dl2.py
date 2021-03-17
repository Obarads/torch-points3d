import os, sys
sys.path.append("../")
import omegaconf
from torch_points3d.datasets.segmentation.s3dis import S3DISOriginalFused

tp3d_dataset_path = "/home/obarads/databox/torch_point3d/"

dataset2 = S3DISOriginalFused(tp3d_dataset_path, test_area=5)
