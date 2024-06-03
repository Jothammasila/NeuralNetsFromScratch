""""
Strictly speaking, convolutional layers are a misnomer, since the operations they
express are more accurately described as cross-correlations. An input tensor and a kernel tensor are
combined to produce an output tensor through a cross-correlation operation.

The output tensor, Y of an imput tensor image X, with the convolution/kernel window K, is given by:

Y = X * K

  where:
        * is the convolution operator.
        X is of shape (n_h, n_w)
            n_h --> the input tensor height.
            n_w --> the input tensor width.
            
        K is of shape(k_h, k_w)
            k_h --> the kernel height.
            k_w --> the kernel  width.
            
        Y is of shape (n_h-k_h+1, n_w-k_w+1)

"""

import torch
from torch import nn

def corr2d(X, K):
    """"Compute 2D cross-correlation"""
    h, w = K.shape
    
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, X):
        return corr2d(X, self.weights) + self.bias