import torch


# 5 10
a = torch.tensor([[1, 1, 2], [3, 4, 5]])
b = torch.tensor([[1, 0, 2], [2, 1, 0]])

c = torch.einsum('ij, ij->i', a, b)
#print(c.shape)