from torch_geometric.data import Data
import torch


td = Data(torch.arange(0,12).reshape(4, 3))
print(td)
print(td[0])
