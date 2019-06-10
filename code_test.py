import torch
from torch import nn
import numpy as np

a = torch.randn(32, 512, 11)
b = torch.randn(32, 512)
a = a.view(a.size(0), -1)
b = b.view(b.size(0), -1)
a = a.cuda()
b = b.cuda()
c = torch.cat([a, b], 1).view(32, 512, -1)
print(np.shape(c))