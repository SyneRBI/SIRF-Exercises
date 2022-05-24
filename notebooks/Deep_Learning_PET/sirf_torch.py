# Simple PyTorch wrapper for primal and dual SIRF operators

# Author: Imraj Singh and Riccardo Barbano

# First version: 21st of May 2022

# CCP SyneRBI Synergistic Image Reconstruction Framework (SIRF).
# Copyright 2022 University College London.

# This is software developed for the Collaborative Computational Project in Synergistic Reconstruction for Biomedical Imaging (http://www.ccpsynerbi.ac.uk/).

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# Based on https://github.com/educating-dip/pet_deep_image_prior/blob/main/src/deep_image_prior/torch_wrapper.py

import torch

class _primal_op(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, image_template, sinogram_template, sirf_obj):
        ctx.sirf_obj = sirf_obj
        ctx.image_template = image_template
        ctx.sinogram_template = sinogram_template
        x_np = x.detach().cpu().numpy()
        x_np = ctx.image_template.fill(x_np)
        proj_sinogram_np = ctx.sirf_obj.forward(x_np).as_array()
        proj_sinogram = torch.from_numpy(proj_sinogram_np).requires_grad_().to(x.device)
        return proj_sinogram.float()

    @staticmethod
    def backward(ctx, sinogram):
        sinogram_np = sinogram.detach().cpu().numpy()
        sinogram_np = ctx.sinogram_template.fill(sinogram_np)
        grads_np = ctx.sirf_obj.backward(sinogram_np).as_array()
        grads = torch.from_numpy(grads_np).requires_grad_().to(sinogram.device)
        return grads.float(), None, None, None, None

class primal_op(torch.nn.Module):
    def __init__(self, image_template, sinogram_template, acq_model):
        super().__init__()
        self.image_template = image_template
        self.sinogram_template = sinogram_template
        self.acq_model = acq_model

    def forward(self, image):
        # x.shape: (N, C, H, W) or (N, C, D, H, W)
        image_nc_flat = image.view(-1, *image.shape[2:])
        sinogram_nc_flat = []
        for x_i in image_nc_flat:
            sym_sinogram_i = _primal_op.apply(x_i.unsqueeze(0), self.image_template, self.sinogram_template, self.acq_model)
            sinogram_nc_flat.append(sym_sinogram_i)
        sinogram = torch.cat(sinogram_nc_flat)
        return sinogram

class _dual_op(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, sinogram, image_template, sinogram_template, sirf_obj):
        ctx.sirf_obj = sirf_obj
        ctx.image_template = image_template
        ctx.sinogram_template = sinogram_template
        sinogram_np = sinogram.detach().cpu().numpy()
        sinogram_np = ctx.sinogram_template.fill(sinogram_np)
        grads_np = ctx.sirf_obj.backward(sinogram_np).as_array()
        grads = torch.from_numpy(grads_np).requires_grad_().to(sinogram.device)
        return grads.float()
        
    @staticmethod
    def backward(ctx, x):
        x_np = x.detach().cpu().numpy()
        x_np = ctx.image_template.fill(x_np)
        proj_sinogram_np = ctx.sirf_obj.forward(x_np).as_array()
        proj_sinogram = torch.from_numpy(proj_sinogram_np).requires_grad_().to(x.device)
        return proj_sinogram.float(), None, None, None, None

class dual_op(torch.nn.Module):
    def __init__(self, image_template, sinogram_template, acq_model):
        super().__init__()
        self.image_template = image_template
        self.sinogram_template = sinogram_template
        self.acq_model = acq_model

    def forward(self, sinogram):
        # x.shape: (N, C, H, W) or (N, C, D, H, W)
        sinogram_nc_flat = sinogram.view(-1, *sinogram.shape[2:])
        image_nc_flat = []
        for x_i in sinogram_nc_flat:
            sym_image_i = _dual_op.apply(x_i[None][None], self.image_template, self.sinogram_template, self.acq_model)
            image_nc_flat.append(sym_image_i)
        image = torch.cat(image_nc_flat).unsqueeze_(dim=1)
        return image