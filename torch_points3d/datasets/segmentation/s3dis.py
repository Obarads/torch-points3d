import enum
import os
import os.path as osp
from itertools import repeat, product
from time import process_time
import numpy as np
import h5py
from numpy.core.shape_base import block
from requests import structures
import torch
import random
import glob
from plyfile import PlyData, PlyElement
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data, extract_zip, Dataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data import DataLoader
from torch_geometric.datasets import S3DIS as S3DIS1x1
import torch_geometric.transforms as T
import logging
from sklearn.neighbors import NearestNeighbors, KDTree
from tqdm.auto import tqdm as tq
import csv
import pandas as pd
import pickle
import gdown
import shutil

from tqdm.std import tqdm

from torch_points3d.datasets.samplers import BalancedRandomSampler
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

S3DIS_NUM_CLASSES = 13

INV_OBJECT_LABEL = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "chair",
    8: "table",
    9: "bookcase",
    10: "sofa",
    11: "board",
    12: "clutter",
}

OBJECT_COLOR = np.asarray(
    [
        [233, 229, 107],  # 'ceiling' .-> .yellow
        [95, 156, 196],  # 'floor' .-> . blue
        [179, 116, 81],  # 'wall'  ->  brown
        [241, 149, 131],  # 'beam'  ->  salmon
        [81, 163, 148],  # 'column'  ->  bluegreen
        [77, 174, 84],  # 'window'  ->  bright green
        [108, 135, 75],  # 'door'   ->  dark green
        [41, 49, 101],  # 'chair'  ->  darkblue
        [79, 79, 76],  # 'table'  ->  dark grey
        [223, 52, 52],  # 'bookcase'  ->  red
        [89, 47, 95],  # 'sofa'  ->  purple
        [81, 109, 114],  # 'board'   ->  grey
        [233, 233, 229],  # 'clutter'  ->  light grey
        [0, 0, 0],  # unlabelled .->. black
    ]
)

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

ROOM_TYPES = {
    "conferenceRoom": 0,
    "copyRoom": 1,
    "hallway": 2,
    "office": 3,
    "pantry": 4,
    "WC": 5,
    "auditorium": 6,
    "storage": 7,
    "lounge": 8,
    "lobby": 9,
    "openspace": 10,
}

VALIDATION_ROOMS = [
    "hallway_1",
    "hallway_6",
    "hallway_11",
    "office_1",
    "office_6",
    "office_11",
    "office_16",
    "office_21",
    "office_26",
    "office_31",
    "office_36",
    "WC_2",
    "storage_1",
    "storage_5",
    "conferenceRoom_2",
    "auditorium_1",
]

################################### UTILS #######################################


def object_name_to_label(object_class):
    """convert from object name in S3DIS to an int"""
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["clutter"])
    return object_label


def read_s3dis_format(train_file, room_name, label_out=True, verbose=False, debug=False):
    """extract data from a room folder"""

    room_type = room_name.split("_")[0]
    room_label = ROOM_TYPES[room_type]
    raw_path = osp.join(train_file, "{}.txt".format(room_name))
    if debug:
        reader = pd.read_csv(raw_path, delimiter="\n")
        RECOMMENDED = 6
        for idx, row in enumerate(reader.values):
            row = row[0].split(" ")
            if len(row) != RECOMMENDED:
                log.info("1: {} row {}: {}".format(raw_path, idx, row))

            try:
                for r in row:
                    r = float(r)
            except:
                log.info("2: {} row {}: {}".format(raw_path, idx, row))

        return True
    else:
        room_ver = pd.read_csv(raw_path, sep=" ", header=None).values
        xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype="float32")
        try:
            rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype="uint8")
        except ValueError:
            rgb = np.zeros((room_ver.shape[0], 3), dtype="uint8")
            log.warning("WARN - corrupted rgb data for file %s" % raw_path)
        if not label_out:
            return xyz, rgb
        n_ver = len(room_ver)
        del room_ver
        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(xyz)
        semantic_labels = np.zeros((n_ver,), dtype="int64")
        room_label = np.asarray([room_label])
        instance_labels = np.zeros((n_ver,), dtype="int64")
        objects = glob.glob(osp.join(train_file, "Annotations/*.txt"))
        i_object = 1
        for single_object in objects:
            object_name = os.path.splitext(os.path.basename(single_object))[0]
            if verbose:
                log.debug("adding object " + str(i_object) + " : " + object_name)
            object_class = object_name.split("_")[0]
            object_label = object_name_to_label(object_class)
            obj_ver = pd.read_csv(single_object, sep=" ", header=None).values
            _, obj_ind = nn.kneighbors(obj_ver[:, 0:3])
            semantic_labels[obj_ind] = object_label
            instance_labels[obj_ind] = i_object
            i_object = i_object + 1

        return (
            torch.from_numpy(xyz),
            torch.from_numpy(rgb),
            torch.from_numpy(semantic_labels),
            torch.from_numpy(instance_labels),
            torch.from_numpy(room_label),
        )


def to_ply(pos, label, file):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    colors = OBJECT_COLOR[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    el = PlyElement.describe(ply_array, "S3DIS")
    PlyData([el], byte_order=">").write(file)

def t2n(tensor):
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

def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        # fix range bug for python3 (warry): range(N)->list(range(N))
        return np.concatenate([data, dup_data], 0), list(range(N))+list(sample) 

def sample_data_label(data, label, inslabel, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    new_inslabel = inslabel[sample_indices]
    return new_data, new_label, new_inslabel

# Ref: https://github.com/WXinlong/ASIS/blob/d71c3d60e985f5bebe620c8e1a1cb0042fb2f5f6/indoor3d_util.py#L137
# Split single room data into blocks.
# room2blocks: Not implementation with torch.
# B = number of blocks in single room.
# points : (B, N, 6) xyzrgb
# sem_labels : (B, N) semantic labels
# ins_labels : (B, N) instance labels
def room2blocks(data, label, inslabel, num_point, block_size=1.0, stride=1.0,
                random_sample=False, sample_num=None, sample_aug=1):
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters
        stride: float, stride for block sweeping
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels
        block_labels: K x num_point x 1 np array of uint8 labels
        
    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert(stride<=block_size)

    limit = np.amax(data, 0)[0:3]
     
    # Get the corner location for our sampling blocks    
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        for i in range(num_block_x):
            if i % 2 == 0:
                for j in range(num_block_y):
                    xbeg_list.append(i*stride)
                    ybeg_list.append(j*stride)
            else:
                for j in range(num_block_y)[::-1]:
                    xbeg_list.append(i*stride)
                    ybeg_list.append(j*stride)

    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0]) 
            ybeg = np.random.uniform(-block_size, limit[1]) 
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)

    # Collect blocks
    block_data_list = []
    block_label_list = []
    block_inslabel_list = []
    idx = 0
    for idx in range(len(xbeg_list)): 
       xbeg = xbeg_list[idx]
       ybeg = ybeg_list[idx]
       xcond = (data[:,0]<=xbeg+block_size) & (data[:,0]>=xbeg)
       ycond = (data[:,1]<=ybeg+block_size) & (data[:,1]>=ybeg)
       cond = xcond & ycond
       if np.sum(cond) < 100: # discard block if there are less than 100 pts.
           continue
       
       block_data = data[cond, :]
       block_label = label[cond]
       block_inslabel = inslabel[cond]
       
       # randomly subsample data
       block_data_sampled, block_label_sampled, block_inslabel_sampled = \
           sample_data_label(block_data, block_label, block_inslabel, num_point)
       block_data_list.append(np.expand_dims(block_data_sampled, 0))
       block_label_list.append(np.expand_dims(block_label_sampled, 0))
       block_inslabel_list.append(np.expand_dims(block_inslabel_sampled, 0))
            
    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0),\
           np.concatenate(block_inslabel_list, 0)

# def room2blocks_plus_normalized(data_label, num_point, block_size, stride,
#                                 random_sample, sample_num, sample_aug):
#     """ room2block, with input filename and RGB preprocessing.
#         for each block centralize XYZ, add normalized XYZ as 678 channels
#     """
#     data = data_label[:,0:6]
#     data[:,3:6] /= 255.0
#     label = data_label[:,-2].astype(np.uint8)
#     inslabel = data_label[:,-1].astype(np.uint8)
def room2blocks_plus_normalized(data, label, inslabel, num_point, block_size=1.0, stride=1.0,
                random_sample=False, sample_num=None, sample_aug=1):
    """
    Parameters
    ----------
    data: np.ndarry (N, 6)
        xyzrgb points in a room
    label: np.ndarray (N)
        semantic labels in a room
    inslabel: np.ndarray (N)
        instance labels in a room
    num_point: int
        number of points in a block
    block_size: float
        block size
    stride: float
        stride for block sweeping
    random_sample: bool
        if True, we will randomly sample blocks in the room
    sample_num: int 
        if random sample, how many blocks to sample [default: None]
    sample_aug: 
        if random sample, how much aug

    Returns
    -------
    new_data_batch: np.ndarray (K, num_point, 9) 
        np array of XYZRGBnormalizedXYZ, RGB is in [0,1]
    label_batch: np.ndarray (K, num_point) 
        np array of uint8 labels
    inslabel_batch: np.ndarray (K, num_point) 
        np array of uint8 labels
    """

    max_room_x = max(data[:,0])
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])
    
    data_batch, label_batch, inslabel_batch = room2blocks(
        data, label, inslabel, num_point, block_size, stride, random_sample, 
        sample_num, sample_aug)

    new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 6] = data_batch[b, :, 0]/max_room_x
        new_data_batch[b, :, 7] = data_batch[b, :, 1]/max_room_y
        new_data_batch[b, :, 8] = data_batch[b, :, 2]/max_room_z
        minx = min(data_batch[b, :, 0])
        miny = min(data_batch[b, :, 1])
        data_batch[b, :, 0] -= (minx+block_size/2)
        data_batch[b, :, 1] -= (miny+block_size/2)
    new_data_batch[:, :, 0:6] = data_batch
    return new_data_batch, label_batch, inslabel_batch

################################### 1m cylinder s3dis ###################################


class S3DIS1x1Dataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        pre_transform = self.pre_transform
        self.train_dataset = S3DIS1x1(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=True,
            pre_transform=self.pre_transform,
            transform=self.train_transform,
        )
        self.test_dataset = S3DIS1x1(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=False,
            pre_transform=pre_transform,
            transform=self.test_transform,
        )
        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)


################################### Used for fused s3dis radius sphere ###################################


class S3DISOriginalFused(InMemoryDataset):
    """ Original S3DIS dataset. Each area is loaded individually and can be processed using a pre_collate transform. 
    This transform can be used for example to fuse the area into a single space and split it into 
    spheres or smaller regions. If no fusion is applied, each element in the dataset is a single room by default.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 6 that denotes the area used for testing
    split: str
        can be one of train, trainval, val or test
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    pre_transform
    transform
    pre_filter
    """

    form_url = (
        "https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1"
    )
    download_url = "https://drive.google.com/uc?id=0BweDykwS9vIobkVPN0wzRzFwTDg&export=download"
    zip_name = "Stanford3dDataset_v1.2_Version.zip"
    path_file = osp.join(DIR, "s3dis.patch")
    file_name = "Stanford3dDataset_v1.2"
    folders = ["Area_{}".format(i) for i in range(1, 7)]
    num_classes = S3DIS_NUM_CLASSES

    def __init__(
        self,
        root,
        test_area=6,
        split="train",
        transform=None,
        pre_transform=None,
        pre_collate_transform=None,
        pre_filter=None,
        keep_instance=False,
        verbose=False,
        debug=False,
    ):
        assert test_area >= 1 and test_area <= 6
        self.transform = transform
        self.pre_collate_transform = pre_collate_transform
        self.test_area = test_area
        self.keep_instance = keep_instance
        self.verbose = verbose
        self.debug = debug
        self._split = split
        super(S3DISOriginalFused, self).__init__(root, transform, pre_transform, pre_filter)
        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        elif split == "trainval":
            path = self.processed_paths[3]
        else:
            raise ValueError((f"Split {split} found, but expected either " "train, val, trainval or test"))
        self._load_data(path)

        if split == "test":
            self.raw_test_data = torch.load(self.raw_areas_paths[test_area - 1])

    @property
    def center_labels(self):
        if hasattr(self.data, "center_label"):
            return self.data.center_label
        else:
            return None

    @property
    def raw_file_names(self):
        return self.folders

    @property
    def pre_processed_path(self):
        pre_processed_file_names = "preprocessed.pt"
        return os.path.join(self.processed_dir, pre_processed_file_names)

    @property
    def raw_areas_paths(self):
        return [os.path.join(self.processed_dir, "raw_area_%i.pt" % i) for i in range(6)]

    @property
    def processed_file_names(self):
        test_area = self.test_area
        return (
            ["{}_{}.pt".format(s, test_area) for s in ["train", "val", "test", "trainval"]]
            + self.raw_areas_paths
            + [self.pre_processed_path]
        )

    @property
    def raw_test_data(self):
        return self._raw_test_data

    @raw_test_data.setter
    def raw_test_data(self, value):
        self._raw_test_data = value

    def download(self):
        raw_folders = os.listdir(self.raw_dir)
        if len(raw_folders) == 0:
            if not os.path.exists(osp.join(self.root, self.zip_name)):
                log.info("WARNING: You are downloading S3DIS dataset")
                log.info("Please, register yourself by filling up the form at {}".format(self.form_url))
                log.info("***")
                log.info(
                    "Press any key to continue, or CTRL-C to exit. By continuing, you confirm filling up the form."
                )
                input("")
                gdown.download(self.download_url, osp.join(self.root, self.zip_name), quiet=False)
            extract_zip(os.path.join(self.root, self.zip_name), self.root)
            shutil.rmtree(self.raw_dir)
            os.rename(osp.join(self.root, self.file_name), self.raw_dir)
            shutil.copy(self.path_file, self.raw_dir)
            cmd = "patch -ruN -p0 -d  {} < {}".format(self.raw_dir, osp.join(self.raw_dir, "s3dis.patch"))
            os.system(cmd)
        else:
            intersection = len(set(self.folders).intersection(set(raw_folders)))
            if intersection != 6:
                shutil.rmtree(self.raw_dir)
                os.makedirs(self.raw_dir)
                self.download()

    def process(self):
        if not os.path.exists(self.pre_processed_path):
            train_areas = [f for f in self.folders if str(self.test_area) not in f]
            test_areas = [f for f in self.folders if str(self.test_area) in f]

            train_files = [
                (f, room_name, osp.join(self.raw_dir, f, room_name))
                for f in train_areas
                for room_name in os.listdir(osp.join(self.raw_dir, f))
                if os.path.isdir(osp.join(self.raw_dir, f, room_name))
            ]

            test_files = [
                (f, room_name, osp.join(self.raw_dir, f, room_name))
                for f in test_areas
                for room_name in os.listdir(osp.join(self.raw_dir, f))
                if os.path.isdir(osp.join(self.raw_dir, f, room_name))
            ]

            # Gather data per area
            data_list = [[] for _ in range(6)]
            if self.debug:
                areas = np.zeros(7)
            for (area, room_name, file_path) in tq(train_files + test_files):
                if self.debug:
                    area_idx = int(area.split("_")[-1])
                    if areas[area_idx] == 5:
                        continue
                    else:
                        print(area_idx)
                        areas[area_idx] += 1

                area_num = int(area[-1]) - 1
                if self.debug:
                    read_s3dis_format(file_path, room_name, label_out=True, verbose=self.verbose, debug=self.debug)
                    continue
                else:
                    xyz, rgb, semantic_labels, instance_labels, room_label = read_s3dis_format(
                        file_path, room_name, label_out=True, verbose=self.verbose, debug=self.debug
                    )

                    rgb_norm = rgb.float() / 255.0
                    data = Data(pos=xyz, y=semantic_labels, rgb=rgb_norm)
                    if room_name in VALIDATION_ROOMS:
                        data.validation_set = True
                    else:
                        data.validation_set = False

                    if self.keep_instance:
                        data.instance_labels = instance_labels

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    data_list[area_num].append(data)

            raw_areas = cT.PointCloudFusion()(data_list)
            for i, area in enumerate(raw_areas):
                torch.save(area, self.raw_areas_paths[i])

            for area_datas in data_list:
                # Apply pre_transform
                if self.pre_transform is not None:
                    for data in area_datas:
                        data = self.pre_transform(data)
            torch.save(data_list, self.pre_processed_path)
        else:
            data_list = torch.load(self.pre_processed_path)

        if self.debug:
            return

        train_data_list = {}
        val_data_list = {}
        trainval_data_list = {}
        for i in range(6):
            if i != self.test_area - 1:
                train_data_list[i] = []
                val_data_list[i] = []
                for data in data_list[i]:
                    validation_set = data.validation_set
                    del data.validation_set
                    if validation_set:
                        val_data_list[i].append(data)
                    else:
                        train_data_list[i].append(data)
                trainval_data_list[i] = val_data_list[i] + train_data_list[i]

        train_data_list = list(train_data_list.values())
        val_data_list = list(val_data_list.values())
        trainval_data_list = list(trainval_data_list.values())
        test_data_list = data_list[self.test_area - 1]

        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            train_data_list = self.pre_collate_transform(train_data_list)
            val_data_list = self.pre_collate_transform(val_data_list)
            test_data_list = self.pre_collate_transform(test_data_list)
            trainval_data_list = self.pre_collate_transform(trainval_data_list)

        self._save_data(train_data_list, val_data_list, test_data_list, trainval_data_list)

    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(val_data_list), self.processed_paths[1])
        torch.save(self.collate(test_data_list), self.processed_paths[2])
        torch.save(self.collate(trainval_data_list), self.processed_paths[3])

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)


class S3DISSphere(S3DISOriginalFused):
    """ Small variation of S3DISOriginalFused that allows random sampling of spheres 
    within an Area during training and validation. Spheres have a radius of 2m. If sample_per_epoch is not specified, spheres
    are taken on a 2m grid.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 6 that denotes the area used for testing
    train: bool
        Is this a train split or not
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    sample_per_epoch
        Number of spheres that are randomly sampled at each epoch (-1 for fixed grid)
    radius
        radius of each sphere
    pre_transform
    transform
    pre_filter
    """

    def __init__(self, root, sample_per_epoch=100, radius=2, *args, **kwargs):
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid_sphere_sampling = cT.GridSampling3D(size=radius / 10.0)
        super().__init__(root, *args, **kwargs)

    def __len__(self):
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return len(self._test_spheres)

    def get(self, idx):
        if self._sample_per_epoch > 0:
            return self._get_random()
        else:
            return self._test_spheres[idx].clone()

    def process(self):  # We have to include this method, otherwise the parent class skips processing
        super().process()

    def download(self):  # We have to include this method, otherwise the parent class skips download
        super().download()

    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_data = self._datas[centre[3].int()]
        sphere_sampler = cT.SphereSampling(self._radius, centre[:3], align_origin=False)
        return sphere_sampler(area_data)

    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(train_data_list, self.processed_paths[0])
        torch.save(val_data_list, self.processed_paths[1])
        torch.save(test_data_list, self.processed_paths[2])
        torch.save(trainval_data_list, self.processed_paths[3])

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            for i, data in enumerate(self._datas):
                assert not hasattr(
                    data, cT.SphereSampling.KDTREE_KEY
                )  # Just to make we don't have some out of date data in there
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos), leaf_size=10)
                setattr(data, cT.SphereSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)


class S3DISCylinder(S3DISSphere):
    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_data = self._datas[centre[3].int()]
        cylinder_sampler = cT.CylinderSampling(self._radius, centre[:3], align_origin=False)
        return cylinder_sampler(area_data)

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            for i, data in enumerate(self._datas):
                assert not hasattr(
                    data, cT.CylinderSampling.KDTREE_KEY
                )  # Just to make we don't have some out of date data in there
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=10)
                setattr(data, cT.CylinderSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridCylinderSampling(self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)


class S3DISFusedDataset(BaseDataset):
    """ Wrapper around S3DISSphere that creates train and test datasets.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - fold: test_area parameter
            - pre_collate_transform
            - train_transforms
            - test_transforms
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        sampling_format = dataset_opt.get("sampling_format", "sphere")
        dataset_cls = S3DISCylinder if sampling_format == "cylinder" else S3DISSphere

        self.train_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=3000,
            test_area=self.dataset_opt.fold,
            split="train",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
        )

        self.val_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="val",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
        )
        self.test_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="test",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
        )

        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save s3dis predictions to disk using s3dis color scheme

        Parameters
        ----------
        pos : torch.Tensor
            tensor that contains the positions of the points
        label : torch.Tensor
            predicted label
        file : string
            Save location
        """
        to_ply(pos, label, file)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.s3dis_tracker import S3DISTracker

        return S3DISTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)


class S3DIS1x1Ins(InMemoryDataset):
    """New dataset
    """

    form_url = (
        "https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1"
    )
    download_url = "https://drive.google.com/uc?id=0BweDykwS9vIobkVPN0wzRzFwTDg&export=download"
    zip_name = "Stanford3dDataset_v1.2_Version.zip"
    path_file = osp.join(DIR, "s3dis.patch")
    file_name = "Stanford3dDataset_v1.2"
    folders = ["Area_{}".format(i) for i in range(1, 7)]
    num_classes = S3DIS_NUM_CLASSES

    num_points = 4096
    block_size = 1.0
    stride = 0.5
    valid_slice_types = ['block', 'room']

    def __init__(self,
                 root,
                 test_area=6,
                 train=True,
                 structure={'train': 'block', 'test': 'room'},
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        assert test_area >= 1 and test_area <= 6
        self.test_area = test_area
        assert structure['train'] in self.valid_slice_types
        assert structure['test'] in self.valid_slice_types
        self.structure = structure

        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]

        self.data, slices = torch.load(path)
        block_slices, room_slices = slices
        if train:
            slice_type = self.structure['train']
        else:
            slice_type = self.structure['test']

        if slice_type == 'block':
            self.slices = block_slices
        elif slice_type == 'room':
            self.slices = room_slices
        else:
            raise ValueError('slice_type is any of these: {}'.format(self.valid_slice_types))

    @property
    def raw_file_names(self):
        return self.folders

    @property
    def processed_file_names(self):
        test_area = self.test_area
        return ['{}_{}.pt'.format(s, test_area) for s in ['train', 'test']]

    def download(self):
        raw_folders = os.listdir(self.raw_dir)
        if len(raw_folders) == 0:
            if not os.path.exists(osp.join(self.root, self.zip_name)):
                log.info("WARNING: You are downloading S3DIS dataset")
                log.info("Please, register yourself by filling up the form at {}".format(self.form_url))
                log.info("***")
                log.info(
                    "Press any key to continue, or CTRL-C to exit. By continuing, you confirm filling up the form."
                )
                input("")
                gdown.download(self.download_url, osp.join(self.root, self.zip_name), quiet=False)
            extract_zip(os.path.join(self.root, self.zip_name), self.root)
            shutil.rmtree(self.raw_dir)
            os.rename(osp.join(self.root, self.file_name), self.raw_dir)
            shutil.copy(self.path_file, self.raw_dir)
            cmd = "patch -ruN -p0 -d  {} < {}".format(self.raw_dir, osp.join(self.raw_dir, "s3dis.patch"))
            os.system(cmd)
        else:
            intersection = len(set(self.folders).intersection(set(raw_folders)))
            if intersection != 6:
                shutil.rmtree(self.raw_dir)
                os.makedirs(self.raw_dir)
                self.download()

    def collate(self, data_list, room_slices):
        data, block_slices = super().collate(data_list)

        new_room_slices = {}
        for key in block_slices:
            new_room_slices[key] = torch.tensor(room_slices)

        return data, block_slices, new_room_slices

    def process(self):
        areas = self.folders
        # Create pt data that include block data of room
        area_data_list = [{} for _ in range(6)]  # [area, {room: [points, sem_labels, ins_labels]}]
        for area in areas:
            area_num = int(area[-1]) - 1
            pt_path = osp.join(self.processed_dir, area + '.pt')

            ### Check area pt file
            if os.path.exists(pt_path):  ### If Area_X.pt is found
                area_data_list[area_num] = torch.load(pt_path)
                print("Load preprocessed data ({}.pt)".format(area))
            else:  ### If Area_X.pt is not found
                ### Get file paths of Area_X
                print("Create preprocessed data ({}.pt)".format(area))
                file_paths = [
                    (area, room_name, osp.join(self.raw_dir, area, room_name))
                    for room_name in os.listdir(osp.join(self.raw_dir, area))
                    if os.path.isdir(osp.join(self.raw_dir, area, room_name))
                ]

                ### Create room data
                for (area, room_name, file_path) in file_paths:
                    ###### extract single room data
                    xyz, rgb, semantic_labels, instance_labels, room_label = \
                        read_s3dis_format(file_path, room_name, label_out=True)
                    rgb_norm = rgb.float() / 255.0

                    ###### create block data (block = 1x1m region)
                    ### Localize
                    xyz_min = torch.min(xyz, dim=0)[0]
                    modified_xyz = xyz - xyz_min

                    ### Split single room data into blocks.
                    points, sem_labels, ins_labels = room2blocks_plus_normalized(
                        t2n(torch.cat([modified_xyz, rgb_norm], dim=1)), 
                        t2n(semantic_labels), t2n(instance_labels), self.num_points, 
                        block_size=self.block_size, stride=self.stride
                    )

                    ### add room data to area data list
                    data = [points, sem_labels, ins_labels]
                    area_data_list[area_num][room_name] = data

                torch.save(area_data_list[area_num], pt_path)
                print("Finished creating preprocessed data ({}.pt)".format(area))

        train_list = []
        test_list = []

        area_room_idx = 0 # room ID
        train_room_slices = [0]
        test_room_slices = [0]
        print("Create dataset ({} and {})".format(self.processed_paths[0], self.processed_paths[1]))
        for area_num, area_data in enumerate(area_data_list):
            if area_num+1 == self.test_area:
                data_list = test_list
                room_slices= test_room_slices
            else:
                data_list = train_list
                room_slices = train_room_slices

            for room_key in area_data:
                points, sem_labels, ins_labels = area_data[room_key]
                check = len(data_list)
                num_slice = 0
                for block_idx in range(len(points)):
                    num_points = len(points[block_idx, :, :3])
                    data = Data(
                        pos=torch.from_numpy(points[block_idx, :, :3]),
                        x=torch.from_numpy(points[block_idx, :, 3:]),
                        y=torch.from_numpy(sem_labels[block_idx]),
                        ins_y=torch.from_numpy(ins_labels[block_idx]),
                        area_room=torch.tensor([area_room_idx]*num_points),
                        block=torch.tensor([block_idx]*num_points)
                    )
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)
                    num_slice += data.pos.shape[0]

                if len(data_list) > check:
                    area_room_idx += 1
                    room_slices.append(room_slices[-1]+num_slice)

        train_data, train_block_slices, train_room_slices = self.collate(train_list, train_room_slices)
        test_data, test_block_slices, test_room_slices = self.collate(test_list, test_room_slices)

        torch.save((train_data, (train_block_slices, train_room_slices)), self.processed_paths[0])
        torch.save((test_data, (test_block_slices, test_room_slices)), self.processed_paths[1])

        print("Finished creating dataset ({} and {})".format(self.processed_paths[0], self.processed_paths[1]))


class S3DIS1x1InsDataset(BaseDataset):
    """Create 1m x 1m block datasets with instance labels from room data of S3DIS.
    """
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        self.train_dataset = S3DIS1x1Ins(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=True,
            structure=self.dataset_opt.structure,
            transform=self.train_transform,
        )
        self.test_dataset = S3DIS1x1Ins(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=False,
            structure=self.dataset_opt.structure,
            transform=self.train_transform,
        )

        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
