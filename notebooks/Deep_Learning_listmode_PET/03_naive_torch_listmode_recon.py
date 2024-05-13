##TODO: x_t with FOV mask, test with EM update (multiply with image)

# %% [markdown]
# Learning objectives
# ===================
#
# 1. How to implement a custom pytorch layer that computes the gradient of the Poisson log likelihood objective function.
# 2. How to implement a custom pytorch network that unrolls the gradient asscent algorithm for the Poisson log likelihood 
#    objective function combined with a pass through a convolutional network with trainable weights.

# %% [markdown]
# Import modules and define file names
# ------------------------------------

# %%
import sirf.STIR
import torch
import numpy as np
from pathlib import Path
from sirf.Utilities import examples_data_path

data_path: Path = Path(examples_data_path("PET")) / "mMR"
list_file: str = str(data_path / "list.l.hdr")
nxny: tuple = (25, 25)
num_subsets: int = 21

# engine's messages go to files, except error messages, which go to stdout
_ = sirf.STIR.MessageRedirector("info.txt", "warn.txt")

# %% [markdown]
# Load listmode data and create the acquisition model
# ---------------------------------------------------
#
# In this demo example, we use a simplified acquisition model that only implements the geometric forward projection.
# The effects of normalization, attenuation, scatter, randoms, are ignored but can be added as shown in the last
# example.

# %%
sirf.STIR.AcquisitionData.set_storage_scheme("memory")
listmode_data = sirf.STIR.ListmodeData(list_file)
acq_data_template = listmode_data.acquisition_data_template()
print(acq_data_template.get_info())

# select acquisition model that implements the geometric
# forward projection by a ray tracing matrix multiplication
acq_model = sirf.STIR.AcquisitionModelUsingRayTracingMatrix()
acq_model.set_num_tangential_LORs(1)

# ==============================================================
# In a real world scenario, the effects of attenuation, scatter, 
# randoms, normalization should be added here.
# We ignore them for the sake of simplicity in this example.
# ==============================================================

# setup an initial (template) image based on the acquisition data template 
initial_image = acq_data_template.create_uniform_image(value=1, xy=nxny)

# %% [markdown]
# Setup of the Poisson log likelihood listmode objective function
# ---------------------------------------------------------------
#
# Using the listmode data and the acquisition model, we can now setup the Poisson log likelihood objective function.

# %%
lm_obj_fun = (
    sirf.STIR.PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin()
)
lm_obj_fun.set_acquisition_model(acq_model)
lm_obj_fun.set_acquisition_data(listmode_data)
lm_obj_fun.set_num_subsets(num_subsets)
print('setting up listmode objective function ...')
lm_obj_fun.set_up(initial_image)

# %% [markdown]
# Setup of a pytorch layer that computes the gradient of the Poisson log likelihood objective function
# ----------------------------------------------------------------------------------------------------
#
# Using our listmode objective function, we can now implement a custom pytorch layer that computes the gradient 
# of the Poisson log likelihood using the `gradient()` method using a subset of the listmode data.
#
# This layer maps a torch minibatch tensor to another torch tensor of the same shape.
# The shape of the minibatch tensor is [batch_size=1, channel_size=1, spatial dimensions]. 
# For the implementation we subclass `torch.nn.Module` and implement the `forward()` method by 
# first converting the torch tensor to sirf.STIR.ImageData, then computing the gradient using the `gradient()` method
# and converting back to a torch tensor.

# %%

class PoissonlogLGradientLayer(torch.nn.Module):  
    def __init__(self, obj_fun, template_image: sirf.STIR.ImageData, subset: int = 0) -> None:
        """Poisson logL Gradient layer

        Parameters
        ----------
        obj_fun : sirf.STIR objective function (listmode or sinogram)
            the objective function to compute the gradient
        template_image : sirf.STIR.ImageData
            template image needed for torch to ImageData conversion
        subset : int, optional
            the subset of the listmode data to use, by default 0
        """        
        super().__init__()
        self._obj_fun = obj_fun
        self._template_image: sirf.STIR.ImageData = template_image.clone()
        self._subset: int = subset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # convert torch tensor to numpy array and strip the batch and channel dimensions
        x_np: np.ndarray = x.numpy(force = True)[0,0,...]
        x_sirf = self._template_image.clone()
        x_sirf.fill(x_np)
        g: sirf.STIR.ImageData = self._obj_fun.gradient(x_sirf, subset=self._subset)

        # convert to back to torch tensor and add batch and channel dimensions
        y = torch.tensor(g.as_array(), device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)

        return y

# %% [markdown]
# Time to test the forward pass of our custom listmode Poisson logL layer

# %%
# seed all random number generators to make the script deterministic
torch.manual_seed(0)
# choose our device
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

# set the batch size
batch_size = 1

# setup a test input mini-batch of images that we use to test our network
# the dimension are [batch, channel, spatial dimensions]
x_t = torch.rand(
    (batch_size, 1) + initial_image.shape,
    device=dev,
    dtype=torch.float32,
    requires_grad=False,
)

# setup the Poisson logL gradient layer
grad_layer = PoissonlogLGradientLayer(lm_obj_fun, initial_image, subset=0)
# pass the input mini-batch through the layer
y_t = grad_layer(x_t)

# %% [markdown]
# Exercise 3.1
# ------------
#
# Using the PoissonGradient layer above and the template code of the next cell, implement the minimal unrolled 
# network shown in the figure below.

# %%
class UnrolledPoissonlogLGradientNet(torch.nn.Module):  
    def __init__(self, obj_fun, template_image: sirf.STIR.ImageData, device: str) -> None:
        """unrolled Poisson logL Gradient network

        Parameters
        ----------
        obj_fun : sirf.STIR objective function (listmode or sinogram)
            the objective function to compute the gradient
        template_image : sirf.STIR.ImageData
            template image needed for torch to ImageData conversion
        subsets : list[int]
            the subsets of the listmode data to use in the blocks of the unrolled network
        device : str
            the device to run the network on
        """        
        super().__init__()
        self._poisson_logL_grad_layer0 = PoissonlogLGradientLayer(obj_fun, template_image, subset=0)
        self._poisson_logL_grad_layer1 = PoissonlogLGradientLayer(obj_fun, template_image, subset=1)
        self._poisson_logL_grad_layer2 = PoissonlogLGradientLayer(obj_fun, template_image, subset=2)

        self._fov_mask = torch.tensor(lm_obj_fun.get_subset_sensitivity(0).as_array() > 0, dtype = torch.float32, device = device).unsqueeze(0).unsqueeze(0)

        # define a minimal convolutional network consisting of a single 3x3x3 convolutional layer
        self._conv_net0 = torch.nn.Conv3d(1, 1, (3,3,3), dtype=torch.float32, device = device, padding='same')
        self._conv_net1 = torch.nn.Conv3d(1, 1, (3,3,3), dtype=torch.float32, device = device, padding='same')
        self._conv_net2 = torch.nn.Conv3d(1, 1, (3,3,3), dtype=torch.float32, device = device, padding='same')
        self._relu = torch.nn.ReLU()

        self._step_size = 1e-5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of the unrolled network

        Parameters
        ----------
        x : torch.Tensor
            input mini-batch of images

        Returns
        -------
        torch.Tensor
            output mini-batch of images. every layer in the network is applied is a combination of a 
            Poisson logL gradient step and a pass through a convolutional network.
        """        
        y0 = self._fov_mask * self._relu(x + self._step_size * self._poisson_logL_grad_layer0(x) + self._conv_net0(x))
        y1 = self._fov_mask * self._relu(y0 + self._step_size * self._poisson_logL_grad_layer1(y0) + self._conv_net1(y0))
        y2 = self._fov_mask * self._relu(y1 + self._step_size * self._poisson_logL_grad_layer2(y1) + self._conv_net2(y1))
        return y2

# %% 
# uncomment the next line and run this cell to see the solution
# %load snippets/solution_3_1.py

# %%
# define the unrolled network
my_net = UnrolledPoissonlogLGradientNet(lm_obj_fun, initial_image, device = dev)
# pass the input mini-batch through the network
prediction = my_net(x_t)


# %% [markdown]
# Loss computation and gradient backpropagation
# ---------------------------------------------
#
# During training, we need to compute the loss and backpropagate the gradient with respect 
# to the trainable parameters (e.g. the parameter of our mini CNN) through the network.
# In the following cell we define a dummy mean squared error loss and compute its values
# using the prediction of our network and the target tensor (here: a dummy tensor full of ones,
# but in real applications, this would contain high-quality target images).

# %%
# setup a dummy mini batch of target images
target = torch.ones_like(x_t, requires_grad=False)
# setup the MSE loss function
loss_fct = torch.nn.MSELoss()
# calculate the MSE loss between the prediction and the target
loss = loss_fct(prediction, target)
# backpropagate the gradient of the loss through the network
# (needed to update the trainable parameters of the network with an optimizer)
loss.backward()

## %% [markdown]
# Exercise 3.2
# ------------
#
# Using our "naive" implementation of our unrolled network `.backward()` will raise an error,
# saying that the gradient cannot be back propagated. Why is that? 
