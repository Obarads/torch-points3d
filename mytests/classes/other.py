import torch
from torch_points3d.core.spatial_ops.neighbour_finder import BaseNeighbourFinder
from torch_cluster import knn

class DenseKNNNeighbourFinder(BaseNeighbourFinder):
    """"
    Parameters
    ----------
    k : int
        Number of Neighbours.
    """
    def __init__(self, k):
        super().__init__()
        # self.knn = KNNNeighbourFinder(k)
        self.k = k
    
    def __call__(self, x, y):
        """KNN for B x N x C

        Parameters
        ----------
        x : torch.tensor
            Features of all points. (B,N,C)
        y : torch.tesnor
            Features of center points. (B,M,C)

        Return:
        indexes : torch.tensor
            K neighbour indexes. (B, M, k)
        """
        return self.find_neighbours(x, y)

    def find_neighbours(self, x, y):
        # B, N, C = x.shape
        with torch.no_grad():
            row_x = self.batch_to_row(x)
            row_y = self.batch_to_row(y)
            batch_x = self.create_batch_num(x)
            batch_y = self.create_batch_num(y)
            # indices = self.knn(row_x, row_y, batch_x, batch_y)
            _, indices = knn(row_x, row_y, self.k, batch_x, batch_y)
            # indices = indices.view()
            indices = self.row_to_batch(indices, *y.shape[:2])
            indices = self.sub_idx_base(indices, x)

        return indices

    def sub_idx_base(self, indices, inputs):
        B, N, _ = inputs.shape
        idx_base = torch.arange(0, B, device=indices.device).view(-1, *[1]*(len(indices.shape)-1)) * N # if len(idx_shape) = 3, .view(-1, 1, 1)
        return indices - idx_base

    def batch_to_row(self, inputs):
        B, N, C = inputs.shape
        inputs = inputs.view(B*N, C)
        return inputs
    
    def row_to_batch(self, inputs, B, N):
        inputs = inputs.view(B, N, self.k)
        return inputs

    def create_batch_num(self, inputs):
        B, N, _ = inputs.shape
        device = inputs.device
        batch_num = torch.arange(0, B, device=device).view(B, 1).repeat(1, N)
        batch_num =  batch_num.view(B*N) 
        return batch_num
