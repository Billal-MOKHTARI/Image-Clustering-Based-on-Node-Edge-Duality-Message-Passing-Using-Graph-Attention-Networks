from torch import nn
import torch

class MeanCosineDistance(nn.Module):
    def __init__(self, dim=1):
        super(MeanCosineDistance, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=dim)

    def forward(self, x, y):
        m = x.size(0)

        mean_value = self.cos_sim(x, y).mean()
        res = 1-mean_value

        return res
    
# criterion = MeanCosineDistance()
# x = torch.tensor([[0, 1], [0, 1]], dtype=torch.float64)
# y = torch.tensor([[1, 0], [1, 0]], dtype=torch.float64)

# res = criterion(x, y)
# print(res)