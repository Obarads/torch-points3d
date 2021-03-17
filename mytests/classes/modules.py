import torch
from torch import nn

from torch_points3d.core.base_conv import BaseConvolution
from torch_points3d.core.base_conv.dense import DenseFPModule

# This class is based on `from torch_points3d.core.base_conv.dense import BaseDenseConvolutionUp`
class DenseDualFPModule(BaseConvolution):
    def __init__(self, up_conv_nn, bn=True, bias=False, activation=nn.ReLU(), **kwargs):
        super(DenseDualFPModule, self).__init__(None, None, **kwargs)

        self.sem_decoder_layer = DenseFPModule(up_conv_nn, bn, bias, activation=activation)
        self.ins_decoder_layer = DenseFPModule(up_conv_nn, bn, bias, activation=activation)

    def forward(self, data, **kwargs):
        """ Propagates features from one layer to the next.
        data contains information from the down convs in data_skip

        Arguments:
            data -- (data, data_skip)
        """
        data, data_skip = data
        sem_data, ins_data = data
        # sem_pos, sem_x = sem_data.pos, sem_data.x
        # pos_skip, x_skip = data_skip.pos, data_skip.x

        sem_data = self.sem_decoder_layer((sem_data, data_skip))
        ins_data = self.ins_decoder_layer((ins_data, data_skip))

        data = (sem_data, ins_data)

        return data

from torch_points3d.core.common_modules.base_modules import BaseModule
import torch_points_kernels as tp
from other import DenseKNNNeighbourFinder

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
        nn_idx = self.neighbour_finder(e_ins, e_ins)
        k_f_sem = tp.grouping_operation(f_sem, nn_idx)
        f_isem = torch.max(k_f_sem, dim=3)[0]
        # f_isem = torch.squeeze(f_isem, dim=3)
        p_sem = self.sem_pred_fc(f_isem)

        return p_sem, e_ins

