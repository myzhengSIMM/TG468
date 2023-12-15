import torch
import torch.nn as nn
from typing import Any
from torch import Tensor
import math
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__( self, input_dim, \
                        output_dim, \
                        support, \
                        act_func = None, \
                        featureless = False, \
                        dropout_rate = 0., \
                        bias=True):
        super(GraphConvolution, self).__init__()
        self.support = support
        self.featureless = featureless
        for i in range(len(self.support)):
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.randn(input_dim, output_dim))) 
        self.bias = bias
        if self.bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))
        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()  
    def reset_parameters(self):   
        nn.init.xavier_uniform_(getattr(self, 'W{}'.format(0)), gain=1)     
        if self.bias:
            nn.init.zeros_(self.b) 
    def forward(self, x):
        x = self.dropout(x)
        for i in range(len(self.support)):
            if self.featureless:  
                pre_sup = getattr(self, 'W{}'.format(i))
                self.weight = pre_sup
            else:
                pre_sup = x.mm(getattr(self, 'W{}'.format(i)))   
                self.weight = getattr(self, 'W{}'.format(i))                
            if i == 0:
                out = self.support[i].mm(pre_sup)
            else: 
                out += self.support[i].mm(pre_sup)
        out = out + self.b
        if self.act_func is not None:
            out = self.act_func(out)
        self.embedding = out        
        return out

class GCN(nn.Module):
    def __init__( self, input_dim, \
                        support,\
                        dropout_rate=0., \
                        num_classes=10):
        super(GCN, self).__init__()
        self.layer1 = GraphConvolution(input_dim, 200, support, act_func=nn.ReLU(), featureless=True, dropout_rate=dropout_rate)
        self.layer2= GraphConvolution(200, 1, support, act_func=nn.Sigmoid(),dropout_rate=dropout_rate)    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
