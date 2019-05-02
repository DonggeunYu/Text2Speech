import torch
from torch.autograd import Variable
import numpy as np
a = np.array((2))
a = torch.LongTensor(a)
b = Variable(a)