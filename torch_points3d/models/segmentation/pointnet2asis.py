import logging
from numpy.lib.shape_base import tile

from scipy import stats
import math
from copy import deepcopy
import time

import torch
from torch import nn

from torch_geometric.data import Data

from torch_points3d.models.base_architectures import UnetBasedModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.datasets.batch import SimpleBatch

from torch_points3d.modules.ASIS import *
from torch_points3d.modules.pointnet2 import *
from torch_points3d.core.losses import DiscriminativeLoss

log = logging.getLogger(__name__)

# def sparse_to_dense_with_idx_list(data:Data, idx_list:torch.tensor):
#     """convert sparse to dense data along index list. (In order to create block data from an area data)
#     If assign 0 ~ B (index) to each data element, return a Data list with B length. The list have Data stacked according to assigned number (index).
#     Args:
#         data (Data): sparse data (N, C)
#         idx_list (torch.tensor): index list (N)
#     Return:
#         data_list: dense data (B, ...)
#     """
#     ids = torch.unique(idx_list, dtype=torch.long)
#     dense = []
#     for id in ids:
#         data_idx = idx_list  == id
#         dense.append(Data[data_idx])
#     return dense

def gaussian(d:torch.FloatTensor, bw:float) -> torch.FloatTensor:
    return torch.exp(-0.5*((d/bw))**2) / (bw*math.sqrt(2*math.pi))

def flat(d:torch.FloatTensor, bw:float) -> torch.FloatTensor:
    res: torch.BoolTensor = d < bw
    return res.to(dtype=d.dtype)

class TorchMeanShift:
    """https://github.com/fastai/courses/blob/master/deeplearning2/meanshift.ipynb
    """

    supported_kernels = {
        'gaussian': gaussian,
        'flat': flat
    }

    def __init__(self, bandwidth, max_iter=300, kernel='flat') -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.max_iter = max_iter

        if kernel in self.supported_kernels:
            self.kernel = self.supported_kernels[kernel]
        else:
            raise NotImplementedError('Supported kernels are {}, actually {}'.format(self.supported_kernels.keys(), kernel))

        self.labels_ = None
        self.cluster_all = True
        self.n_iter_ = 0

    @staticmethod
    def _get_pairwise_distances(a,b) -> torch.FloatTensor:
        """Get a pairwise distance matrix.
        """
        return torch.sqrt(((a.unsqueeze(0) - b.unsqueeze(1))**2).sum(2)).T

    @staticmethod
    def _get_radius_nn_mask(X1, X2, radius, including_myself=True) -> torch.BoolTensor:
        """Get mask of pairwise distance matrix for radius nearest neighbors.
        Args:
            X1 (torch.tensor) : input (N1, C)
            X2 (torch.tensor) : input (N2, C)
            radius (float): radius
            including_myself (bool): In case of including_myself=False and X1=X2, return results not including myself in radius nearest neighbors. But, when there is no neighborhood, nan may be mixed into radius_nn_mean (return).
        Return:
            radius_nn_mask (torch.tensor): mask of radius nearest neighbors (N1, N2)

        Examples:
            X = torch.arange(12, dtype=torch.float32).reshape(4,3)
            mask = MeanshiftG._get_radius_nn_mask(X, X, 5)
        """
        dist = TorchMeanShift._get_pairwise_distances(X1, X2)
        radius_nn_mask:torch.tensor = dist < radius

        # If X1.shape=X2.shape and including_myself is True, this function does not include myself in raidus NN for the mean calculation.
        # But, when there is no neighborhood, nan may be mixed.
        if not including_myself and X1 == X2:
            N, C = X1.shape
            diag_mask = torch.diag(torch.ones(N)).to(dtype=torch.bool)
            radius_nn_mask[diag_mask] = False

        return radius_nn_mask

    @staticmethod
    def _get_radius_nn_mean(X:torch.tensor, radius:float, including_myself=True) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        """Get mean of nearest neighbors in radius.
        Args:
            X (torch.tensor) : inputs (N, C)
            radius (float): radius
            including_myself (bool): In case of False, return results not including myself in radius nearest neighbors. But, when there is no neighborhood, nan may be mixed into radius_nn_mean (return).
        Return:
            radius_nn_mean (torch.tensor): mean of radius nearest neighbors (N)
            radius_nn_mask (torch.tensor): mask of radius nearest neighbors (N, N)

        Examples:
            X = torch.arange(12, dtype=torch.float32).reshape(4,3)
            tms = TorchMeanShift(5, max_iter=5)
            mean_data, mask = tms._mean_radius_nn(X)
        """

        # Get radius NN mask
        radius_nn_mask = TorchMeanShift._get_radius_nn_mask(X, X, radius, including_myself)

        # Get radius NN mean (including myself)
        N, C = X.shape
        X = torch.tile(X[:, None, :], (1, N, 1))
        X = X.transpose(0, 1)
        X[radius_nn_mask == False] = 0
        radius_nn_mean:torch.FloatTensor = X.sum(1) / radius_nn_mask.sum(-1)[:, None]

        return radius_nn_mean, radius_nn_mask
    
    @staticmethod
    def _t2n(torch_tensor:torch.Tensor) -> np.ndarray:
        """torch.Tensor to numpy.ndarray
        """
        return torch_tensor.detach().cpu().numpy()

    def _create_labels(self, X:torch.tensor, original_X:torch.tensor):
        device = X.device

        # get all_res (sklearn)
        radius_nn_mean, radius_nn_mask = self._get_radius_nn_mean(X, self.bandwidth)
        num_nn = torch.sum(radius_nn_mask, dim=1)
        all_res = [(tuple(radius_nn_mean[i].cpu().numpy().tolist()), num_nn[i]) for i in range(len(radius_nn_mean))]

        seeds = original_X
        center_intensity_dict = {}
        for i in range(len(seeds)):
            if all_res[i][1]:  # i.e. len(points_within) > 0
                center_intensity_dict[all_res[i][0]] = all_res[i][1]

        sorted_by_intensity = sorted(center_intensity_dict.items(),
                                     key=lambda tup: (tup[1], tup[0]),
                                     reverse=True)
        sorted_centers = torch.tensor([tup[0] for tup in sorted_by_intensity], device=device)

        radius_nn_mask = TorchMeanShift._get_radius_nn_mask(sorted_centers, sorted_centers, self.bandwidth)
        unique = torch.ones(len(sorted_centers), dtype=bool, device=device)

        for i in range(len(sorted_centers)):
            if unique[i]:
                neighbor_idxs = radius_nn_mask[i]
                unique[neighbor_idxs] = 0
                unique[i] = 1  # leave the current point as unique
        cluster_centers = sorted_centers[unique]

        # ASSIGN LABELS: a point belongs to the cluster that it is closest to
        dist = self._get_pairwise_distances(original_X, cluster_centers)
        idxs = torch.argmin(dist, dim=1)
        labels = torch.zeros(len(original_X), dtype=int)

        if self.cluster_all:
            labels = idxs.flatten()
        else:
            raise NotImplementedError()
            # labels.fill(-1)
            # bool_selector = dist.flatten() <= self.bandwidth
            # labels[bool_selector] = idxs.flatten()[bool_selector]

        self.cluster_centers_ = TorchMeanShift._t2n(cluster_centers)
        self.labels_ = TorchMeanShift._t2n(labels)

    def fit(self, X:torch.FloatTensor):
        original_X = deepcopy(X)
        stop_thresh = 1e-3 * self.bandwidth

        with torch.no_grad():
            # X = torch.FloatTensor(np.copy(X)).cuda()
            for it in range(self.max_iter):
                weight = self.kernel(self._get_pairwise_distances(X, X), self.bandwidth)
                num = (weight[:, :, None] * X).sum(1)
                X_new = num / weight.sum(1)[:, None]

                # check convergence
                shift = torch.abs(X_new - X).sum()/torch.abs(original_X.sum())
                X = X_new
                self.n_iter_ += 1
                if shift < stop_thresh:
                    break

            self._create_labels(X, original_X)

    def predict(self):
        raise NotImplementedError()

# from sklearn.cluster import MeanShift
MeanShift = TorchMeanShift

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
        self.labels: torch.tensor = data.y.to(device)

        # ins labels
        self.ins_labels: torch.tensor = data.ins_y.to(device)

        # self.batch_idx = torch.arange(0, data.pos.shape[0]).view(-1, 1).repeat(1, data.pos.shape[1]).view(-1)
        if self._use_category:
            self.category = data.categor
        
        # Get room ID
        if self.model.training:
            self.room_id_list = data.area_room

    def forward(self, epoch=-1, *args, **kwargs):
        output, embed_ins = self._network(self.input)
        self.output = output.transpose(1,2) # (B, NUM_CLASSES, N) -> (B, N, NUM_CLASSES) for Tracker
        self.embed_ins = embed_ins # (B, ins_output_dim, N)

        if not self.model.training:  # for eval.py
            ins_output_labels = []
            ins_seg_list = []

            sem_output_labels = torch.argmax(self.output, dim=-1) # Get prediction semantic labels (B, N, NUM_CLASSES) -> (B, N)
            sem_output_labels = sem_output_labels.cpu().detach().numpy() # torch to numpy
            # embed_inses = self.embed_ins.cpu().detach().numpy() # torch to numpy
            embed_inses = self.embed_ins

            for block_idx in range(len(embed_inses)):
                pred_sem_label = sem_output_labels[block_idx]
                embed_ins = embed_inses[block_idx]

                start = time.time()
                num_clusters, pred_ins_label, cluster_centers = \
                    self._cluster(embed_ins)
                print(time.time()-start)
                start = time.time()
                ins_seg = {}
                for idx_cluster in range(num_clusters):
                    tmp = (pred_ins_label == idx_cluster)
                    if np.sum(tmp) != 0: # add (for a cluster of zero element.)
                        a = stats.mode(pred_sem_label[tmp])[0]
                        estimated_seg = int(a)
                        ins_seg[idx_cluster] = estimated_seg
                print(time.time()-start)
                ins_output_labels.append(pred_ins_label)
                ins_seg_list.append(ins_seg)

                self.ins_output_labels = ins_output_labels
                self.ins_seg_list = ins_seg_list

        return self.output

    def _network(self, input):
        data = self.model(input)
        sem_data, ins_data = data

        sem_data = self.sem_layer(sem_data.x)
        ins_data = self.ins_layer(ins_data.x)
        pred_sem, embed_ins = self.asis(sem_data, ins_data)

        return pred_sem, embed_ins

    def get_output_for_BlockMerging(self):
        return self.ins_output_labels, self.ins_seg_list, self.room_id_list

    def get_labels_for_BlockMerging(self):
        return self.ins_labels

    def _cluster(self, embeddings):
        ms = MeanShift(bandwidth=self.opt.cluster.bandwidth)
        ms.fit(embeddings)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        num_clusters = cluster_centers.shape[0]

        return num_clusters, labels, cluster_centers

    def _compute_loss(self):
        self.semantic_loss = self.cross_entropy_loss(
            self.output.transpose(1, 2),
            self.labels
        )
        self.loss = self.opt.loss.cross_entropy_loss.loss_weights * \
            self.semantic_loss

        # Instance loss
        self.instance_loss = self.discriminative_loss(
            self.embed_ins,
            self.ins_labels
        )
        self.loss += self.opt.loss.discriminative_loss.loss_weights * \
            self.instance_loss

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self._compute_loss()
        self.loss.backward()

    def _dump_visuals(self, epoch):
        return
