import logging
import random
import os
from sklearn.cluster import MeanShift

import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.data import Data

# from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.models.base_architectures import UnetBasedModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL

# from .structures import PanopticLabels, PanopticResults
from torch_points3d.modules.ASIS import *
from torch_points3d.modules.pointnet2 import *
from torch_points3d.core.losses import DiscriminativeLoss

log = logging.getLogger(__name__)


class PointNet2ASIS(UnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        UnetBasedModel.__init__(self, option, model_type, dataset, modules)

        # Extract parameters from the dataset
        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes
        self._use_category = getattr(option, "use_category", False)
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

    def set_input(self, data, device):
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
        data = data.to(device)

        # pos and features
        assert len(data.pos.shape) == 3
        if data.x is not None:
            x = data.x.transpose(1, 2).contiguous()
        else:
            x = None
        self.input = Data(x=x, pos=data.pos)

        # sem labels
        if data.y is not None:
            self.gt_sem_labels = data.y  # [B * N]
        else:
            self.gt_sem_labels = None

        # ins labels
        if data.instance_labels is not None:
            self.gt_ins_labels = data.instance_labels
        else:
            self.gt_ins_labels = None

        data = data.to(device)

        # self.batch_idx = torch.arange(0, data.pos.shape[0]).view(-1, 1).repeat(1, data.pos.shape[1]).view(-1)
        if self._use_category:
            self.category = data.category

    def forward(self, epoch=-1, *args, **kwargs):
        data = self.model(self.input)

        sem_data, ins_data = data

        # Use ASIS.
        sem_data = self.sem_layer(sem_data.x)
        ins_data = self.ins_layer(ins_data.x)
        self.pred_sem, self.embed_ins = self.asis(sem_data, ins_data)

        if self.training():
            with torch.no_grad():
                # self._dump_visuals(epoch)
                num_clusters, labels, cluster_centers = self._cluster(self.embed_ins)

        self.output = self.pred_sem.transpose(1,2) # (B, C, N) -> (B, N, C) for Tracker
        self.labels = self.gt_sem_labels

        return self.output

    def _cluster(self, embeddings):
        ms = MeanShift(bandwidth=self.opt.bandwidth, bin_seeding=True)

        #print ('Mean shift clustering, might take some time ...')
        #tic = time.time()
        ms.fit(embeddings)
        #print ('time for clustering', time.time() - tic)
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
        if random.random() < self.opt.vizual_ratio:
            if not hasattr(self, "visual_count"):
                self.visual_count = 0
            data_visual = Data(
                pos=self.raw_pos, y=self.input.y, instance_labels=self.input.instance_labels, batch=self.input.batch
            )
            data_visual.semantic_pred = torch.max(self.output.semantic_logits, -1)[1]
            data_visual.vote = self.output.offset_logits
            nms_idx = self.output.get_instances()
            if self.output.clusters is not None:
                data_visual.clusters = [self.output.clusters[i].cpu() for i in nms_idx]
                data_visual.cluster_type = self.output.cluster_type[nms_idx]
            if not os.path.exists("viz"):
                os.mkdir("viz")
            torch.save(data_visual.to("cpu"), "viz/data_e%i_%i.pt" % (epoch, self.visual_count))
            self.visual_count += 1
