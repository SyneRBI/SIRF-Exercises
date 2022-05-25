"""
Taken (and lightly modified) from https://github.com/cetmann/pytorch-primaldual

MIT License

Copyright (c) 2020 cetmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Simple implementation of the learned primal-dual approach by 
Adler & Öktem (2017), https://arxiv.org/abs/1707.06474
"""


import torch
import torch.nn as nn
from sirf_torch import primal_op, dual_op


class ConcatenateLayer(nn.Module):
    def __init__(self):
        super(ConcatenateLayer, self).__init__()
    
    def forward(self, *x):
        return torch.cat(list(x), dim=1)

class DualNet(nn.Module):
    def __init__(self, n_dual, n_layers, n_feature_channels):
        super(DualNet, self).__init__()
        
        self.n_dual = n_dual
        self.n_channels = n_dual + 2
        self.n_layers = n_layers
        self.n_feature_channels = n_feature_channels
        
        self.input_concat_layer = ConcatenateLayer()
        
        layers = [nn.Conv2d(self.n_channels, self.n_feature_channels, kernel_size=3, padding=1),]
        for _ in range(self.n_layers):
            layers.append(nn.PReLU())
            layers.append(nn.Conv2d(self.n_feature_channels, self.n_feature_channels, kernel_size=3, padding=1))
        layers.append(nn.PReLU())
        layers.append( nn.Conv2d(self.n_feature_channels, self.n_dual, kernel_size=3, padding=1))
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, h, Op_f, g):
        x = self.input_concat_layer(h, Op_f, g)
        x = h + self.block(x)
        return x
    
class PrimalNet(nn.Module):
    def __init__(self, n_primal, n_layers, n_feature_channels):
        super(PrimalNet, self).__init__()
        
        self.n_primal = n_primal
        self.n_channels = n_primal + 1
        self.n_layers = n_layers
        self.n_feature_channels = n_feature_channels
        
        self.input_concat_layer = ConcatenateLayer()
        layers = [nn.Conv2d(self.n_channels, self.n_feature_channels, kernel_size=3, padding=1),]
        for _ in range(self.n_layers):
            layers.append(nn.PReLU())
            layers.append(nn.Conv2d(self.n_feature_channels, self.n_feature_channels, kernel_size=3, padding=1))
        layers.append(nn.PReLU())
        layers.append( nn.Conv2d(self.n_feature_channels, self.n_primal, kernel_size=3, padding=1))
        self.block = nn.Sequential(*layers)
        
    def forward(self, f, OpAdj_h):
        x = self.input_concat_layer(f, OpAdj_h)
        x = f + self.block(x)
        return x
    
class LearnedPrimalDual(nn.Module):
    def __init__(self,
                image_template,
                sinogram_template,
                acq_model,
                primal_architecture = PrimalNet,
                dual_architecture = DualNet,
                n_iter = 10,
                n_primal = 5,
                n_dual = 5,
                n_layers = 5,
                n_feature_channels = 128):
        
        super(LearnedPrimalDual, self).__init__()
        
        self.primal_architecture = primal_architecture
        self.dual_architecture = dual_architecture
        self.n_iter = n_iter
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.n_layers = n_layers
        self.n_feature_channels = n_feature_channels
        
        self.primal_shape = (n_primal,) + image_template.shape[1:]
        self.dual_shape = (n_dual,) + sinogram_template.shape[2:] 
        
        self.primal_op_layer = primal_op(image_template, sinogram_template, acq_model)
        self.dual_op_layer = dual_op(image_template, sinogram_template, acq_model)
        
        self.primal_nets = nn.ModuleList()
        self.dual_nets = nn.ModuleList()
        
        self.concatenate_layer = ConcatenateLayer()
        
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.dirac_(m.weight)
                m.bias.data.fill_(0.0)
        
        for i in range(n_iter):
            self.primal_nets.append(
                primal_architecture(n_primal, n_layers, n_feature_channels)
            )
            self.dual_nets.append(
                dual_architecture(n_dual, n_layers, n_feature_channels)
            )
        self.primal_nets.apply(init_weights)
        self.dual_nets.apply(init_weights)

    def forward(self, g, intermediate_values = False):
        
        h = torch.zeros(g.shape[0:1] + (self.dual_shape), device=g.device)
        f = torch.zeros(g.shape[0:1] + (self.primal_shape), device=g.device)
        
        if intermediate_values:
            h_values = []
            f_values = []
            
        for i in range(self.n_iter):
            ## Dual
            # Apply forward operator to f^(2)
            f_2 = f[:,1:2]
            if intermediate_values:
                f_values.append(f)
            Op_f = self.primal_op_layer(f_2)
            # Apply dual network
            h = self.dual_nets[i](h, Op_f, g)
            
            ## Primal
            # Apply adjoint operator to h^(1)
            h_1 = h[:,0:1]
            if intermediate_values:
                h_values.append(h)
            OpAdj_h = self.dual_op_layer(h_1)
            # Apply primal network
            f = self.primal_nets[i](f, OpAdj_h)
        
        if intermediate_values:
            return f[:,0:1], f_values, h_values

        return f[:,0:1]