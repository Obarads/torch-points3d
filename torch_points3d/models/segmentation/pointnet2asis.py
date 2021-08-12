import logging
import random
import os
import re
from sklearn.cluster import MeanShift
from scipy import stats
from typing import List

import torch
from torch import nn
from torch._C import dtype
from torch.nn import functional as F

from torch_geometric.data import Data

# from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.models.base_architectures import UnetBasedModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.datasets.batch import SimpleBatch

# from .structures import PanopticLabels, PanopticResults
from torch_points3d.modules.ASIS import *
from torch_points3d.modules.pointnet2 import *
from torch_points3d.core.losses import DiscriminativeLoss

log = logging.getLogger(__name__)

class BlcokMerging:
    def __init__(self, num_classes:int, gap:float=5e-3, mean_num_pts_in_group:np.ndarray=None):
        self.num_classes = num_classes
        self.gap = gap
        if mean_num_pts_in_group is None:
            self.mean_num_pts_in_group = np.ones(self.num_classes, dtype=np.float32)
        else:
            self.mean_num_pts_in_group = mean_num_pts_in_group.astype(np.float32)

        self.init_data()
    
    def init_data(self):
        volume_num = int(1. / self.gap)+1
        self.volume = -1* np.ones([volume_num,volume_num,volume_num]).astype(np.int32)
        self.volume_seg = -1* np.ones([volume_num,volume_num,volume_num]).astype(np.int32)

        self.ins_output = []
        self.sem_output = []
        self.points = []

    def append_block(self, block_points:np.ndarray, sem_output:np.ndarray, ins_output:np.ndarray, ins_seg:dict):
        """
        Args:
            block_points (np.ndarray): xyz points (N, C)
            sem_output (np.ndarray): semantic segmentation output (N)
            ins_output (np.ndarray): instance segmentation output (N)
            ins_seg (dict)
        """
        merged_ins_output = self._block_merging(self.volume, self.volume_seg, block_points, ins_output, ins_seg, self.gap)
        self.ins_output.append(merged_ins_output)
        self.sem_output.append(sem_output)
        self.points.append(block_points)

    @staticmethod
    def _block_merging(volume, volume_seg, pts, grouplabel, groupseg, gap=1e-3):
        overlapgroupcounts = np.zeros([100,300])
        groupcounts = np.ones(100)
        x=(pts[:,0]/gap).astype(np.int32)
        y=(pts[:,1]/gap).astype(np.int32)
        z=(pts[:,2]/gap).astype(np.int32)
        for i in range(pts.shape[0]):
            xx=x[i]
            yy=y[i]
            zz=z[i]
            if grouplabel[i] != -1:
                if volume[xx,yy,zz]!=-1 and volume_seg[xx,yy,zz]==groupseg[grouplabel[i]]:
                    #overlapgroupcounts[grouplabel[i],volume[xx,yy,zz]] += 1
                    try:
                        overlapgroupcounts[grouplabel[i],volume[xx,yy,zz]] += 1
                    except:
                        pass
            groupcounts[grouplabel[i]] += 1

        groupcate = np.argmax(overlapgroupcounts,axis=1)
        maxoverlapgroupcounts = np.max(overlapgroupcounts,axis=1)

        curr_max = np.max(volume)
        for i in range(groupcate.shape[0]):
            if maxoverlapgroupcounts[i]<7 and groupcounts[i]>30:
                curr_max += 1
                groupcate[i] = curr_max

        finalgrouplabel = -1 * np.ones(pts.shape[0])

        for i in range(pts.shape[0]):
            if grouplabel[i] != -1 and volume[x[i],y[i],z[i]]==-1:
                volume[x[i],y[i],z[i]] = groupcate[grouplabel[i]]
                volume_seg[x[i],y[i],z[i]] = groupseg[grouplabel[i]]
                finalgrouplabel[i] = groupcate[grouplabel[i]]
        return finalgrouplabel

    def get_result(self) -> np.ndarray:
        scene_pc = self.points.reshape([-1, self.points.shape[-1]])
        scene_ins_label = self.ins_output.reshape(-1)
        volume = self.volume
        x = (scene_pc[:, 6] / self.gap).astype(np.int32)
        y = (scene_pc[:, 7] / self.gap).astype(np.int32)
        z = (scene_pc[:, 8] / self.gap).astype(np.int32)
        for i in range(scene_ins_label.shape[0]):
            if volume[x[i], y[i], z[i]] != -1:
                scene_ins_label[i] = volume[x[i], y[i], z[i]]

        un = np.unique(scene_ins_label)
        pts_in_pred = [[] for itmp in range(self.num_classes)]
        group_pred_final:np.ndarray = -1 * np.ones_like(scene_ins_label)
        grouppred_cnt = 0
        for ig, g in enumerate(un): #each object in prediction
            if g == -1:
                continue
            tmp = (scene_ins_label == g)
            sem_seg_g = int(stats.mode(scene_ins_label[tmp])[0])
            #if np.sum(tmp) > 500:
            if np.sum(tmp) > 0.25 * self.mean_num_pts_in_group[sem_seg_g]:
                group_pred_final[tmp] = grouppred_cnt
                pts_in_pred[sem_seg_g] += [tmp]
                grouppred_cnt += 1
        
        return group_pred_final.astype(np.int32), self.sem_output.astype(np.int32)

def sparse_to_dense_with_idx_list(data:Data, idx_list:torch.tensor):
    """convert sparse to dense data along index list. (In order to create block data from an area data)
    If assign 0 ~ B (index) to each data element, return a Data list with B length. The list have Data stacked according to assigned number (index).
    Args:
        data (Data): sparse data (N, C)
        idx_list (torch.tensor): index list (N)
    Return:
        data_list: dense data (B, ...)
    """
    ids = torch.unique(idx_list, dtype=torch.long)
    dense = []
    for id in ids:
        data_idx = idx_list  == id
        dense.append(Data[data_idx])
    return dense

class PointNet2ASIS(UnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        UnetBasedModel.__init__(self, option, model_type, dataset, modules)

        # Extract parameters from the dataset
        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes
        self._mean_num_pts_in_group = dataset.mean_num_pts_in_group
        self._use_category = getattr(option, "use_category", False)
        self._gap = option.block_merging.gap
        if self._use_category:
            if not dataset.class_to_segments:
                raise ValueError(
                    "The dataset needs to specify a class_to_segments property when using category information for segmentation"
                )
            self._class_to_seg = dataset.class_to_segments
            self._num_categories = len(self._class_to_seg)
            log.info("Using category information for the predictions with %i categories", self._num_categories)
        else:
            self._num_categories = 0

        # Build layers from decoder to ASIS.
        decoder_last_layer_dim = option.up_conv.up_conv_nn[-1][-1]
        asis_opt = option.asis
        self.sem_layer = nn.Sequential(
            nn.Conv1d(decoder_last_layer_dim, asis_opt.input_dim, 1),
            nn.BatchNorm1d(asis_opt.input_dim),
            nn.ReLU()
        )
        self.ins_layer = nn.Sequential(
            nn.Conv1d(decoder_last_layer_dim, asis_opt.input_dim, 1),
            nn.BatchNorm1d(asis_opt.input_dim),
            nn.ReLU()
        )

        # Build ASIS modeule.
        self.asis = ASIS(
            num_sem_in_features=asis_opt.input_dim,
            num_sem_out_features=self._num_classes,
            num_ins_in_features=asis_opt.input_dim,
            num_ins_out_features=asis_opt.ins_output_dim,
            k=asis_opt.k
        )

        # Define Discriminative loss.
        dl_opt = option.loss.discriminative_loss
        self.discriminative_loss = DiscriminativeLoss(
            delta_d=dl_opt.delta_d,
            delta_v=dl_opt.delta_v,
            alpha=dl_opt.var,
            beta=dl_opt.dist,
            gamma=dl_opt.reg,
            norm_p=dl_opt.norm_p
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

        self.loss_names = ["instance_loss", "semantic_loss"]
        self.visual_names = ["data_visual"]
        # self.init_weights()

    def set_input(self, data: SimpleBatch, device: torch.device):
        """
        Parameters
        ----------
        data: torch_geometric.data.Data
            x: torch.tensor (B, N, C)
                Features, C is feature dim other than coordinates.
            pos: torch.tensor (B, N, 3)
                Coordinates.
            y: torch.tensor (B, N)
                Semantic labels.
            ins: torch.tensor (B, N)
                Instance labels.
        device:
            Device info.
        """


        # pos and features
        assert len(data.pos.shape) == 3
        if data.x is not None:
            x = data.x.transpose(1, 2).contiguous().to(torch.float32)
        else:
            x = None
        self.input:Data = Data(x=x, pos=data.pos.to(torch.float32)).to(device)

        # sem labels
        self.gt_sem_labels: torch.tensor = data.y.to(device)

        # ins labels
        self.gt_ins_labels: torch.tensor = data.ins_y.to(device)

        # self.batch_idx = torch.arange(0, data.pos.shape[0]).view(-1, 1).repeat(1, data.pos.shape[1]).view(-1)
        if self._use_category:
            self.category = data.categor

    def forward(self, epoch=-1, *args, **kwargs):
        if epoch == -1 and not self.model.training:  # for eval.py
            assert len(self.input) == 1, 'Please set batch_size==1.'
            block_idxs = self.input.block # Get block index per point.
            block_inputs = sparse_to_dense_with_idx_list(self.input[0], block_idxs) # (N_A, C) -> (N_B, N, C), N_A=N_B * M
            area_bm = BlcokMerging(self._num_classes, self._gap, self._mean_num_pts_in_group)
            for block in block_inputs:
                block = torch.unsqueeze(block, 0) # (N, C) -> (1, N, C), numpy to torch
                pred_sem, embed_ins = self._network(block)

                pred_sem = pred_sem.cpu().detach().numpy()[0] # (1, N, C) -> (N, C), torch to numpy
                embed_ins = embed_ins.cpu().detach().numpy()[0] # (1, N, C) -> (N, C), torch to numpy
                pred_sem_label = np.argmax(pred_sem, axis=1)

                num_clusters, block_pred_ins_label, cluster_centers = \
                    self.cluster(embed_ins, bandwidth=0.6)
                ins_seg = {}
                for idx_cluster in range(num_clusters):
                    tmp = (block_pred_ins_label == idx_cluster)
                    if np.sum(tmp) != 0: # add (for a cluster of zero element.)
                        a = stats.mode(pred_sem_label[tmp])[0]
                        estimated_seg = int(a)
                        ins_seg[idx_cluster] = estimated_seg
                area_bm.append_block(block, pred_sem, block_pred_ins_label, ins_seg)
            ins_output, sem_output = area_bm.get_result()
            self.ins_output = ins_output
            self.ins_labels = self.gt_ins_labels
            self.output = sem_output
            self.labels = self.gt_sem_labels
        else: # for train.py
            self.pred_sem, self.embed_ins = self._network(self.input)
            self.output = self.pred_sem.transpose(1,2) # (B, C, N) -> (B, N, C) for Tracker
            self.labels = self.gt_sem_labels

        return self.output

    def _network(self, input):
        data = self.model(input)
        sem_data, ins_data = data

        sem_data = self.sem_layer(sem_data.x)
        ins_data = self.ins_layer(ins_data.x)
        pred_sem, embed_ins = self.asis(sem_data, ins_data)

        return pred_sem, embed_ins

    def get_ins_output(self):
        return self.ins_output
    
    def get_ins_labels(self):
        return self.ins_labels

    def _cluster(self, embeddings):
        ms = MeanShift(bandwidth=self.opt.bandwidth, bin_seeding=True)
        ms.fit(embeddings)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        num_clusters = cluster_centers.shape[0]

        return num_clusters, labels, cluster_centers

    def _compute_loss(self):
        # Semantic loss
        # self.semantic_loss = torch.nn.functional.nll_loss(
        #     self.pred_sem, self.gt_sem_labels, ignore_index=IGNORE_LABEL
        # )
        self.semantic_loss = self.cross_entropy_loss(self.pred_sem, 
                                                     self.gt_sem_labels)
        self.loss = self.opt.loss.cross_entropy_loss.loss_weights * \
            self.semantic_loss

        # Instance loss
        self.instance_loss = self.discriminative_loss(
            self.embed_ins,
            self.gt_ins_labels
        )
        self.loss += self.opt.loss.discriminative_loss.loss_weights * \
            self.instance_loss

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self._compute_loss()
        self.loss.backward()

    def _dump_visuals(self, epoch):
        return
