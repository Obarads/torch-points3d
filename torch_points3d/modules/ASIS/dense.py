import math
from copy import deepcopy
from typing import Tuple

import torch
from torch import nn
import torch_points_kernels as tp
from torch_geometric.data import Data

from torch_points3d.core.base_conv import BaseConvolution
from torch_points3d.core.base_conv.dense import DenseFPModule
from torch_points3d.core.common_modules.base_modules import BaseModule
from torch_points3d.core.spatial_ops.neighbour_finder import DenseKNNNeighbourFinder


# This class is based on `BaseDenseConvolutionUp`
class DenseFPModules(BaseConvolution):
    """Parallel FP Module."""

    def __init__(self, up_conv_nn, bn=True, bias=False, activation=nn.ReLU(), **kwargs):
        super(DenseFPModules, self).__init__(None, None, **kwargs)

        # semantic seg. branch layer
        self.sem_decoder_layer = DenseFPModule(up_conv_nn, bn, bias, activation=activation)
        # instance seg. branch layer
        self.ins_decoder_layer = DenseFPModule(up_conv_nn, bn, bias, activation=activation)

    def forward(self, data, **kwargs):
        """
        Args:
            data (tuple): previous layer output and skip data (data_dec, data_skip)

        Note:
            data_dec (Tuple or torch_geometric): If data_dec is a tuple,
                data_dec contains sem_data and ins_data because data_dec is a
                decoder layer output.
                If data_dec is a torch_geometric.data.Data, data_dec is
                last encoder layer output.
            data_skip: skip connection data
        """
        data_dec, data_skip = data
        if isinstance(data_dec, tuple):
            # If data_dec is a tuple, data_dec contains sem_data and ins_data because data_dec is a decoder layer output.
            sem_data = (data_dec[0], data_skip)
            ins_data = (data_dec[1], data_skip)
        else:
            # If data_dec is a torch_geometric.data.Data, data_dec is last encoder layer output.
            assert isinstance(data_dec, Data)
            assert isinstance(data_skip, Data)
            sem_data = (data_dec, data_skip)
            ins_data = (data_dec, data_skip)

        sem_data = self.sem_decoder_layer(sem_data)
        ins_data = self.ins_decoder_layer(ins_data)

        data = (sem_data, ins_data)

        return data


class ASIS(BaseModule):
    def __init__(self, num_sem_in_features, num_sem_out_features, num_ins_in_features, num_ins_out_features, k):
        super(ASIS, self).__init__()

        # sem branch
        # input: F_ISEM, output: P_SEM
        self.sem_pred_fc = nn.Sequential(
            nn.Dropout(inplace=True), nn.Conv1d(num_sem_in_features, num_sem_out_features, 1)
        )

        # Adaptation
        self.adaptation = nn.Sequential(
            nn.Conv1d(num_sem_in_features, num_ins_in_features, 1), nn.BatchNorm1d(num_ins_in_features), nn.ReLU()
        )

        # # ins branch
        # input: F_SINS, output: E_INS
        self.ins_emb_fc = nn.Sequential(
            nn.Dropout(inplace=True), nn.Conv1d(num_ins_in_features, num_ins_out_features, 1)
        )

        # kNN
        self.neighbour_finder = DenseKNNNeighbourFinder(k)
        self.k = k

    def forward(self, f_sem, f_ins):
        adapted_f_sem = self.adaptation(f_sem)

        # for E_INS
        f_sins = f_ins + adapted_f_sem
        e_ins = self.ins_emb_fc(f_sins)

        # for P_SEM
        # (B, C, N) -> (B, N, C)
        e_ins = e_ins.transpose(1, 2).contiguous()
        # get indices (B, N, k)
        nn_idx = self.neighbour_finder(e_ins, e_ins)
        # get knn features (B, C, N, k)
        k_f_sem = tp.grouping_operation(f_sem, nn_idx)
        f_isem = torch.max(k_f_sem, dim=3)[0]
        p_sem = self.sem_pred_fc(f_isem)

        return p_sem, e_ins


def _gaussian(d: torch.FloatTensor, bw: float) -> torch.FloatTensor:
    return torch.exp(-0.5 * ((d / bw)) ** 2) / (bw * math.sqrt(2 * math.pi))


def _flat(d: torch.FloatTensor, bw: float) -> torch.FloatTensor:
    res: torch.BoolTensor = d < bw
    return res.to(dtype=d.dtype)


class TorchMeanShift:
    """https://github.com/fastai/courses/blob/master/deeplearning2/meanshift.ipynb"""

    supported_kernels = {"gaussian": _gaussian, "flat": _flat}

    def __init__(self, bandwidth, max_iter=300, kernel="flat") -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.max_iter = max_iter

        if kernel in self.supported_kernels:
            self.kernel = self.supported_kernels[kernel]
        else:
            raise NotImplementedError(
                "Supported kernels are {}, actually {}".format(self.supported_kernels.keys(), kernel)
            )

        self.labels_ = None
        self.cluster_all = True
        self.n_iter_ = 0

    @staticmethod
    def _get_pairwise_distances(a, b) -> torch.FloatTensor:
        """Get a pairwise distance matrix."""
        return torch.sqrt(((a.unsqueeze(0) - b.unsqueeze(1)) ** 2).sum(2)).T

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
        radius_nn_mask: torch.tensor = dist < radius

        # If X1.shape=X2.shape and including_myself is True, this function does not include myself in raidus NN for the mean calculation.
        # But, when there is no neighborhood, nan may be mixed.
        if not including_myself and X1 == X2:
            N, _ = X1.shape
            diag_mask = torch.diag(torch.ones(N)).to(dtype=torch.bool)
            radius_nn_mask[diag_mask] = False

        return radius_nn_mask

    @staticmethod
    def _get_radius_nn_mean(
        X: torch.tensor, radius: float, including_myself=True
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
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
        radius_nn_mean: torch.FloatTensor = X.sum(1) / radius_nn_mask.sum(-1)[:, None]

        return radius_nn_mean, radius_nn_mask

    def _create_labels(self, X: torch.tensor, original_X: torch.tensor):
        device = X.device

        # get all_res (sklearn)
        radius_nn_mean, radius_nn_mask = self._get_radius_nn_mean(X, self.bandwidth)
        num_nn = torch.sum(radius_nn_mask, dim=1)

        num_nn_mask = num_nn > 0  # i.e. len(points_within) > 0
        num_nn = num_nn[num_nn_mask]
        radius_nn_mean = radius_nn_mean[num_nn_mask]

        num_nn = num_nn[num_nn_mask]
        radius_nn_mean = radius_nn_mean[num_nn_mask]
        centroid_data = torch.cat([num_nn[:, None].to(torch.float32), radius_nn_mean], dim=1)
        unique_centroid_data = torch.unique(centroid_data, dim=0, sorted=True)
        unique_centroid_data = torch.flipud(unique_centroid_data)
        sorted_centers = unique_centroid_data[:, 1:]

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

        self.cluster_centers_ = cluster_centers.detach().cpu().numpy()
        self.labels_ = labels.detach().cpu().numpy()

    def fit(self, X: torch.FloatTensor):
        original_X = deepcopy(X)
        stop_thresh = 1e-3 * self.bandwidth

        with torch.no_grad():
            # X = torch.FloatTensor(np.copy(X)).cuda()
            for it in range(self.max_iter):
                weight = self.kernel(self._get_pairwise_distances(X, X), self.bandwidth)
                num = (weight[:, :, None] * X).sum(1)
                X_new = num / weight.sum(1)[:, None]

                # check convergence
                shift = torch.abs(X_new - X).sum() / torch.abs(original_X.sum())
                X = X_new
                self.n_iter_ += 1
                if shift < stop_thresh:
                    break
            self._create_labels(X, original_X)

    def predict(self):
        raise NotImplementedError()
