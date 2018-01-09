####################################################
##### This is focal loss class for multi class #####
##### University of Tokyo Doi Kento            #####
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CrossEntropy2d(nn.Module):

    def __init__(self, dim=1,  weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropy2d, self).__init__()

        """
        dim             : dimention along which log_softmax will be computed
        weight          : class balancing weight
        size_average    : which size average will be done or not
        ignore_index    : index that ignored while training
        """
        self.dim = dim
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        
        criterion =  nn.NLLLoss2d(self.weight, self.size_average, self.ignore_index)
        return criterion(F.log_softmax(input, dim=self.dim), target)
