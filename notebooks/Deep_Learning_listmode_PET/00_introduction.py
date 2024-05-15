# %% [markdown]
# Introduction
# ============
#
# In this series of SIRF exercises, we will learn how to build and train a deep neural
# network for listmode PET reconstruction. As a concrete example,
# we will focus on unrolled Variational networks that can be trained in a supervised manner.
# The general architecture of such network is shown below.
#
# ![](figs/osem_varnet.drawio.svg)
#
# The aim of an unrolled variational PET listmode network is to create "high quality" PET reconstructions
# from "low-quality" input listmode data.

# %% [markdown]
# Question
# --------
#
# Which (realistic) circumstances can lead to "low-quality" PET listmode data?
# How can we obtain paired "high-quality" PET reconstructions needed for supervised training?

# %% [markdown]
# Learning objectives of this notebook
# ------------------------------------
#
# 1. What is listmode PET reconstruction and why is it attractive for combining DL and reconstruction.
# 2. Understanding architectures of unrolled reconstruction networks.
# 3. Understanding the essential blocks of training a neural network in pytorch (model setup, data loading, gradient backpropagation)
#    and what we are missing from pytorch to build an unrolled PET reconstruction network.

# %% [markdown]
# What is listmode PET reconstruction? Why is it attractive for combining DL and reconstruction?
# ----------------------------------------------------------------------------------------------
#
# In listmode PET reconstruction, the emission data is stored in a list of events. Each event contains
# the detector numbers of the two detectors that detected the photon pair, and eventually also the
# arrival time difference between the two photons (time-of-flight or TOF).
#
# In contrast to histogrammed emission data (singoram mode), reconstruction of listmode data has the following advantages:
# 1. For low and normal count acquisitions with modern TOF scanners, forward and back projections in listmode are usually faster
#    compared to projections in sinogram mode. **Question: Why?**
# 2. Storage of (low count) listmode data requires less memory compared to storing full TOF sinograms. **Question: Why?**
# 3. Listmode data also preserves the timing information of the detected photon pairs.


# %% [markdown]
# Architecture of unrolled reconstruction networks
# ------------------------------------------------
#
# Unrolled variational networks are a class of deep neural networks that are designed to solve inverse problems.
# The consist of a series of layers that are repeated multiple times.
# Each contains an update with respect to the data fidelity term (blue boxes in the figure above)
# and a regularization term (red boxes in the figure above).
# The latter can be represented by a neural network (e.g. a CNN) containing learnable parameters which are optimized
# during (supervised) training.
#
# There are many way of implementing the data fidelity update block.
# One simple possibility is to implement a gradient ascent step with respect to the Poisson log-likelihood.
# $$ x^+ = x_k + \alpha \nabla_x \log L(y|x) ,$$
# where the Poisson log-likelihood is given by
# $$ \log L(y|x) = \sum_{i} y_i \log(\bar{y}_i(x)) - \bar{y}_i(x) ,$$
# where $y$ is the measured emission sinogram, and $\bar{y}(x) = Ax + s$ the expectation of the measured data given the current
# estimate of the image $x$ and a linear (affine) forward model $A$ including the mean of known additive contaminations (randoms and scatter) $s$.
#

# %% [markdown]
# Exercise 0.1
# ------------
#
# Given the equations above, derive the update formula for the gradient of the Poisson log-likelihood (using sinogram data)
#
# (bonus question) How does the update formula change if we use listmode data instead of sinogram data?

# %% [markdown]
# Solution 0.1
# ------------
#
# YOUR SOLUTION GOES IN HERE

# %%
# DO SHOW THE SOLUTION, UNCOMMENT AND RUN THE FOLLOWING LINES
#
# from IPython.display import Markdown, display
# display(Markdown("snippets/solution_0_1.md"))

# %% [markdown]
# Training a neural network in pytorch
# ------------------------------------
#
# Pytorch is a popular deep learning framework that provides a flexible and efficient way to build and train neural networks.
# The essential steps to train a neural network in pytorch are summarized in the train loop, see
# [here](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#optimizing-the-model-parameters) for more details.


# %%
# DO NOT RUN THIS CELL - CODE SNIPPET ONLY
import torch


def train(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    # loop over the dataset and sample mini-batches
    for batch_num, (input_data_batch, target_image_batch) in enumerate(dataloader):
        # move input and target data to device
        input_data_batch = input_data_batch.to(device)
        target_image_batch = target_image_batch.to(device)

        # Compute prediction error
        predicted_image_batch = model(input_data_batch)
        loss = loss_fn(predicted_image_batch, target_image_batch)

        # calculate gradients using backpropagation
        loss.backward()
        # update model parameters
        optimizer.step()
        # reset gradients
        optimizer.zero_grad()


# model and data loader to be defined
my_model = myModel()
my_data_loader = myDataLoader()

# compute device - use cuda GPU if available
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the loss function we optimize during training
my_loss_fn = torch.nn.MSELoss()
# the optimizer we use to update the model parameters
my_optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-3)

# run a single epoch of training
train(my_data_loader, my_model, my_loss_fn, my_optimizer, dev)


# %% [markdown]
# **The essential blocks for supervised training a neural network in pytorch are:**
# 1. Sampling of mini-batches of input and target (label) images from the training dataset.
# 2. Forward pass: Compute the prediction of the model given the input data.
# 3. Compute the loss (error) between the prediction and the target images.
# 4. Backward pass: Compute the gradient of the loss with respect to the model parameters using backpropagation.
# 5. Update the model parameters using an optimizer.
#
# Fortunately, pytorch provides many high-level functions that simplify the implementation of all these steps.
# (e.g. pytorch's data loader classes, pytorch's convolutional layers and non-linear activation function, pytorch's
# autograd functionality for backpropagation of gradients, and optimizers like Adam)
# To train a listmode PET unrolled variational network, the only thing we need to implement ourselves
# is the forward pass of our model, including the data fidelity update blocks which are not directly available pytorch.
#
# **The aim of the remaining exercises is:**
# - to learn how to couple SIRF/STIR's PET listmode classes into a pytorch feedforward model
# - learn how to backpropagate gradients through our custom model
#
# **The following is beyond the scope of the exercises:**
# - training a real world unrolled variational listmode PET reconstruction network on a
#   big amount of data
