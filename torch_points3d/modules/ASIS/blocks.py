import torch
from torch import nn

from torch_geometric.data import Data

from torch_points3d.core.base_conv import BaseConvolution
from torch_points3d.core.base_conv.dense import DenseFPModule

# This class is based on `GlobalBaseModule`
# class CopyModule(torch.nn.Module):
#     def __init__(self, num_copies, *args, **kwargs):
#         super(CopyModule, self).__init__()
#         self.num_copies = num_copies

#     def forward(self, data, **kwargs):
#         copies = [data for i in range(self.num_copies)]
#         return copies
def copy_data(num_copies, data):
    """Return a list of copied data

    Parameters
    ----------
    num_copies : int
        number of copies
    data : torch.geometric.Data
        Data

    Returns
    -------
    copied_data : list
        copied data list
    """
    return [data for i in range(num_copies)]

# This class is based on `BaseDenseConvolutionUp`
class DenseDualFPModule(BaseConvolution):
    def __init__(self, up_conv_nn, bn=True, bias=False, activation=nn.ReLU(), **kwargs):
        super(DenseDualFPModule, self).__init__(None, None, **kwargs)
        
        # semantic seg. branch layer
        self.sem_decoder_layer = DenseFPModule(up_conv_nn, bn, bias, activation=activation)
        # instance seg. branch layer
        self.ins_decoder_layer = DenseFPModule(up_conv_nn, bn, bias, activation=activation)

    def forward(self, data, **kwargs):
        """
        Parameters
        ----------
        data: tuple
            previous layer output and skip data (data_dec, data_skip)
            data_dec: tuple or torch_geometric
                If data_dec is a tuple, data_dec contains sem_data and ins_data because data_dec is a decoder layer output.
                If data_dec is a torch_geometric.data.Data, data_dec is last encoder layer output.
            data_skip:
                skip connection data
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

from torch_points3d.core.common_modules.base_modules import BaseModule
import torch_points_kernels as tp
from torch_points3d.core.spatial_ops.neighbour_finder import DenseKNNNeighbourFinder

class ASIS(BaseModule):
    def __init__(self, num_sem_in_features, num_sem_out_features, 
                 num_ins_in_features, num_ins_out_features, k):
        super(ASIS, self).__init__()

        # # sem branch
        self.sem_pred_fc = nn.Sequential(
            nn.Dropout(inplace=True),
            nn.Conv1d(num_sem_in_features, num_sem_out_features, 1)
        ) # input: F_ISEM, output: P_SEM
        # sem branch
        # self.sem_pred_fc = nn.Sequential(
        #     nn.Dropout(inplace=True),
        #     nn.Linear(num_sem_in_features, num_sem_out_features)
        # ) # input: F_ISEM, output: P_SEM

        # interactive module: sem to ins
        self.adaptation = nn.Sequential(
            nn.Conv1d(num_sem_in_features, num_ins_in_features, 1),
            nn.BatchNorm1d(num_ins_in_features),
            nn.ReLU()
        )
        # self.adaptation = nn.Sequential(
        #     nn.Linear(num_sem_in_features, num_ins_in_features),
        #     nn.BatchNorm1d(num_ins_in_features),
        #     nn.ReLU()
        # )

        # # ins branch
        self.ins_emb_fc = nn.Sequential(
            nn.Dropout(inplace=True),
            nn.Conv1d(num_ins_in_features, num_ins_out_features, 1)
        ) # input: F_SINS, output: E_INS
        # ins branch
        # self.ins_emb_fc = nn.Sequential(
        #     nn.Dropout(inplace=True),
        #     nn.Linear(num_ins_in_features, num_ins_out_features)
        # ) # input: F_SINS, output: E_INS

        # interactive module: ins to sem
        # using knn_index and index2points

        self.neighbour_finder = DenseKNNNeighbourFinder(k)
        self.k = k

    def forward(self, f_sem, f_ins):
        adapted_f_sem = self.adaptation(f_sem)

        # for E_INS
        f_sins = f_ins + adapted_f_sem
        e_ins = self.ins_emb_fc(f_sins)

        # for P_SEM
        # nn_idx, _ = k_nearest_neighbors(e_ins, e_ins, self.k)
        e_ins = e_ins.transpose(1,2).contiguous() # (B, C, N) -> (B, N, C)
        nn_idx = self.neighbour_finder(e_ins, e_ins) # get indices (B, N, k)
        k_f_sem = tp.grouping_operation(f_sem, nn_idx) # get knn features (B, C, N, k)
        f_isem = torch.max(k_f_sem, dim=3)[0]
        # f_isem = torch.squeeze(f_isem, dim=3)
        p_sem = self.sem_pred_fc(f_isem)

        return p_sem, e_ins


