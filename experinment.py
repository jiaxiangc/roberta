import torch

dic = {0: "A", 1: "B", 2: "C", 3: "D"}

a = torch.tensor([[1, 2, 3, 4]])
print(dic[a.max(dim=1).indices.item()])

