# %% [markdown]
# SIRF.STIR ImageData objects vs numpy arrays vs torch tensors
# ============================================================

# %% [markdown]
# Learning objectives of this notebook
# ------------------------------------
#
# 1. Understanding the differences between SIRF ImageData, numpy arrays and torch tensors.
# 2. Learn how to convert between these different data types.


# %% [markdown]
# SIRF.STIR ImageData objects vs numpy arrays
# -------------------------------------------

# %%
# create a SIRF image template

import sirf.STIR
from sirf.Utilities import examples_data_path

# read an example PET acquisition data set that we can use
# to set up a compatible image data set
acq_data: sirf.STIR.AcquisitionData = sirf.STIR.AcquisitionData(
    examples_data_path("PET") + "/brain/template_sinogram.hs"
)

# create a SIRF image compatible with the acquisition data
# uses default voxel sizes and dimensions
sirf_image_1: sirf.STIR.ImageData = acq_data.create_uniform_image(1.0)
sirf_image_2: sirf.STIR.ImageData = acq_data.create_uniform_image(2.0)

image_shape: tuple[int, int, int] = sirf_image_1.shape

print()
print(f"sirf_image_1 shape   .: {sirf_image_1.shape}")
print(f"sirf_image_1 spacing .: {sirf_image_1.spacing}")
print(f"sirf_image_1 max     .: {sirf_image_1.max()}")
print()
print(f"sirf_image_2 shape   .: {sirf_image_2.shape}")
print(f"sirf_image_2 spacing .: {sirf_image_2.spacing}")
print(f"sirf_image_2 max     .: {sirf_image_2.max()}")

# %%
# you retrieve the data behind a SIRF.STIR image as numpy array using the as_array() method
import numpy as np

numpy_image_1: np.ndarray = sirf_image_1.as_array()
numpy_image_2: np.ndarray = sirf_image_2.as_array()

numpy_image_2_modified = numpy_image_2.copy()
numpy_image_2_modified[0, 0, 0] = 5.0
numpy_image_2_modified[-1, -1, -1] = -4.0

print()
print(f"numpy_image_1 shape   .: {numpy_image_1.shape}")
print(f"numpy_image_1 max     .: {numpy_image_1.max()}")
print()
print(f"numpy_image_2 shape   .: {numpy_image_2.shape}")
print(f"numpy_image_2 max     .: {numpy_image_2.max()}")
print()
print(f"numpy_image_2_modified shape .: {numpy_image_2_modified.shape}")
print(f"numpy_image_2_modified max   .: {numpy_image_2_modified.max()}")


# %%
# you can convert a numpy array into a SIRF.STIR image using the fill() method

# create a copy of sirf_image_2
sirf_image_2_modified = sirf_image_2.get_uniform_copy()
sirf_image_2_modified.fill(numpy_image_2_modified)

print()
print(f"sirf_image_2 shape   .: {sirf_image_2.shape}")
print(f"sirf_image_2 spacing .: {sirf_image_2.spacing}")
print(f"sirf_image_2 max     .: {sirf_image_2.max()}")

# %% [markdown]
# Exercise 2.1
# ------------
#
# Create a SIRF.STIR image that is compatible with the acquisition data
# where every image "plane" contains the "plane number squared".


# %%
# uncomment the next line and run this cell
# %load snippets/solution_2_1.py

# %% [markdown]
# torch tensors vs numpy arrays
# -----------------------------

# %%
import torch

# torch tensors can live on different devices
if torch.cuda.is_available():
    # if cuda is availalbe, we want our torch tensor on the first CUDA device
    dev = torch.device("cuda:0")
else:
    # otherwise we select the CPU as device
    dev = torch.device("cpu")

torch_image_1: torch.Tensor = torch.ones(image_shape, dtype=torch.float32, device=dev)

print()
print(f"torch_image_1 shape  .: {torch_image_1.shape}")
print(f"torch_image_1 max    .: {torch_image_1.max()}")
print(f"torch_image_1 dtype  .: {torch_image_1.dtype}")
print(f"torch_image_1 devive .: {torch_image_1.device}")

# %%
# you can convert torch (GPU or CPU) tensors to numpy arrays using numpy() method
numpy_image_from_torch_1: np.ndarray = torch_image_1.cpu().numpy()
# see here: https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html

# Attention: If the torch tensor lives on the CPU, the underlying array is not copied
# and shared between the numpy and torch object!
print()
print(f"numpy data pointer {numpy_image_from_torch_1.ctypes.data}")
print(f"torch data pointer {torch_image_1.data_ptr()}")

if torch_image_1.data_ptr() == numpy_image_from_torch_1.ctypes.data:
    print("numpy array and torch tensor share same data")
else:
    print("numpy array and torch tensor don't share same data")

# %%
# You can create torch tensors from numpy array using torch.from_numpy()
torch_image_from_numpy_1: torch.Tensor = torch.from_numpy(numpy_image_2)
print()
print(f"torch_image_from_numpy_1 shape  .: {torch_image_from_numpy_1.shape}")
print(f"torch_image_from_numpy_1 max    .: {torch_image_from_numpy_1.max()}")

# torch.from_numpy() will create a Tensor living on the CPU
print()
print(f"device of torch tensor from numpy {torch_image_from_numpy_1.device}")

# we can send the tensor to our prefered device using the .to() method
print(f"sending tensor to device {dev.type}")
torch_image_from_numpy_1.to(dev)
print(f"device of torch tensor from numpy {torch_image_from_numpy_1.device}")


# %% [markdown]
# Exercise 2.2
# ------------
#
# Now that we know how to convert between SIRF.STIR images and numpy arrays,
# and between numpy arrays and torch tensors do the following:
# 1. convert a torch tensor full of "3s" into SIRF.STIR ImageData object compatible
#    with the acquisition data
# 2. convert a SIRF.STIR ImageData object "sirf_image_1" into a torch tensor on the
#    device "dev"
# 3. Predict whether the different image objects should share data and test your
#    hypothesis
# 4. Try to convert the torch tensor `torch.ones(image_shape, dtype=torch.float32, device=dev, requires_grad=True)`
#    into a numpy array. What do you observe?
