import torch
from torch import nn
import numpy as np

a = torch.rand(32, 2, 3)
b = nn.Linear(3, 10)
c = b(a)
print(np.shape(c))