import logging
import torch
from sklearn.cluster import MeanShift
import random
import os

from torch import nn
from torch.nn import functional as F

from torch_geometric.data import Data

from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL

from .structures import PanopticLabels, PanopticResults
from .modules import ASIS
from torch_points3d.core.losses import DiscriminativeLoss

log = logging.getLogger(__name__)

class PointNet2ASIS(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
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

        # Assemble encoder / decoder
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)

        # Build layers from decoder to ASIS.
        self.sem_layer = nn.Sequential(
            nn.Linear(1, 1),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        self.ins_layer = nn.Sequential(
            nn.Linear(1, 1),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )

        # Build ASIS modeule.
        last_mlp_opt = option.mlp_cls
        self.asis = ASIS(
            num_sem_in_features=last_mlp_opt.nn[0],
            num_sem_out_features=last_mlp_opt.sem_nn,
            num_ins_in_features=last_mlp_opt.nn[0],
            num_ins_out_features=last_mlp_opt.ins_nn,
            k=last_mlp_opt.k
        )

        # Define Discriminative loss.
        self.discriminative_loss = DiscriminativeLoss(
            delta_d=1,
            delta_v=1,
            alpha=1,
            beta=1,
            gamma=1,
            norm_p=1
        )

        self.loss_names = ["instance_loss", "semantic_loss"]
        self.visual_names = ["data_visual"]
        self.init_weights()

    def set_input(self, data, device):
        self.raw_pos = data.pos.to(device)
        self.input = data
        all_labels = {l: data[l].to(device) for l in self.__REQUIRED_LABELS__}
        self.labels = PanopticLabels(**all_labels) 

    def forward(self, epoch=-1, *args, **kwargs):
        data = self.model(self.input)
        print(data.shape)
        sem_data, ins_data = data

        # Use ASIS.
        sem_data = self.sem_layer(sem_data)
        ins_data = self.ins_layer(ins_data)
        self.pred_sem, self.embed_ins = self.asis(sem_data, ins_data)

        # Grouping and scoring
        # cluster_scores = None
        # all_clusters = None
        # cluster_type = None

        if self.opt.get_pred_ins_label:
            # Sets visual data for debugging
            with torch.no_grad():
                # self._dump_visuals(epoch)
                num_clusters, labels, cluster_centers = self._cluster(self.embed_ins)

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
        self.semantic_loss = torch.nn.functional.nll_loss(
            self.pred_sem, self.labels.y, ignore_index=IGNORE_LABEL
        )
        self.loss = self.opt.loss_weights.semantic * self.semantic_loss

        # Instance loss
        self.instance_loss = self.discriminative_loss(
            self.embed_ins,
            self.labels.instance_labels
        )
        self.loss += self.opt.loss_weights.instance * self.instance_loss

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
