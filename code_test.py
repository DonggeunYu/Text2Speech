import torch
from torch import nn
import numpy as np

from text import text_to_sequence

a = '안녕하세요'
b = '요안'
print(text_to_sequence(a))
print(text_to_sequence(b))