import torch
from torch._C import dtype

X = torch.arange(12, dtype=torch.float32).reshape(4,3)
N = len(X)
# radius_nn_mask = torch.ones((N, N), dtype=torch.float32)
radius_nn_mask = torch.tensor([
    [1, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 1, 1, 1],
    [0, 1, 1, 1],
], dtype=torch.long)
diag_mask = torch.diag(torch.ones(len(X))).to(dtype=torch.bool)
radius_nn_mask = radius_nn_mask.to(dtype=torch.bool)
radius_nn_mask[diag_mask] = False
print(radius_nn_mask)
# radius_nn_maskwc = torch.tile(radius_nn_mask[:, :, None], (1, 1, len(X[0])))

print(X)
XX = X[:, None, :]
XX = torch.tile(XX, (1, len(X), 1))
XX = XX.transpose(0, 1)
XX[radius_nn_mask == False] = 0

a = XX.sum(1)
b = radius_nn_mask.sum(-1)
b = b[:, None]
radius_nn_mean = a / b

print(radius_nn_mean)

