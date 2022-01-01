import torch
from torch import nn
import torch_points_kernels as tp

from torch_geometric.data import Data

from torch_points3d.core.base_conv import BaseConvolution
from torch_points3d.core.base_conv.dense import DenseFPModule
from torch_points3d.core.common_modules.base_modules import BaseModule
from torch_points3d.core.spatial_ops.neighbour_finder import DenseKNNNeighbourFinder


# class CopyModule(torch.nn.Module):
#     def __init__(self, num_copies):
#         super(CopyModule, self).__init__()
#         self.num_copies = num_copies

#     def forward(self, data, **kwargs):
#         return [data for i in range(self.num_copies)]


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
