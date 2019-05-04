import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
input = Variable(torch.zeros(3, 4))

a = torch.randn(1, 10)
c = F.softsign(Variable(a))
print(c)