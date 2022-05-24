# Author: Imraj Singh

# First version: 21st of May 2022

# CCP SyneRBI Synergistic Image Reconstruction Framework (SIRF).
# Copyright 2022 University College London.

# This is software developed for the Collaborative Computational Project in Synergistic Reconstruction for Biomedical Imaging (http://www.ccpsynerbi.ac.uk/).

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


import torch
import numpy as np
from .misc import random_phantom, shepp_logan

class EllipsesDataset(torch.utils.data.Dataset):

    """ Pytorch Dataset for simulated ellipses

    Initialisation
    ----------
    fwd_op : `SIRF acquisition model`
        The forward operator
    image template : `SIRF image data`
        needed to project and to get shape
    n_samples : `int`
        Number of samples    
    mode : `string`
        Type of data: training, validation and testing
    seed : `int`
        The seed used for the random ellipses
    """

    def __init__(self, fwd_op, image_template, n_samples = 100, mode="train", seed = 1):
        self.fwd_op = fwd_op
        self.image_template = image_template
        self.n_samples = n_samples

        if mode == 'valid':
            self.x_gt = shepp_logan(self.image_template.shape)
            self.y = self.__get_measured__(self.x_gt)

        self.primal_op_layer = fwd_op
        self.mode = mode
        np.random.seed(seed)

    def __get_measured__(self, x_gt):
        # Forward project image then add noise
        y = self.fwd_op(self.image_template.fill(x_gt))
        y = np.random.poisson(y.as_array()[0])
        return y

    def __len__(self):
        # Denotes the total number of iters
        return self.n_samples

    def __getitem__(self, index):
        # Generates one sample of data
        if self.mode == "train":
            x_gt = random_phantom(self.image_template.shape)
            y = self.__get_measured__(x_gt)

        elif self.mode == "valid":
            x_gt = self.x_gt
            y = self.y

        else:
            NotImplementedError

        return x_gt, y