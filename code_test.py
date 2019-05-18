from torch import nn
import torch
import numpy as np
import tensorflow as tf

x = torch.randn(3, 4, 2)
x = x.view(-1, 2)
print(np.shape(x))
b_size = x.size(0)
print(np.shape(b_size))
x = nn.Linear(x)
x = x.view(b_size, -1, 1)