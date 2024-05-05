# %% [markdown]
# Introduction
# ============
#
# In this series of SIRF exercises, we will learn how to build and train a deep neural
# network for listmode PET reconstruction. As a concrete example,
# we will focus on unrolled Variational networks that can be trained in a supervised manner.
# The general architecture of such network is shown below.
#
# ![title](figs/varnet.drawio.svg)

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
# There are many way of implementing the data fidelity update term.
# One simple possibility is to implement a gradient ascent step with respect to the Poisson log-likelihood.
# $$ x^+ = x_k + \alpha \nabla_x \log L(y|x) ,$$
# where the Poisson log-likelihood is given by
# $$ \log L(y|x) = \sum_{i} y_i \log(\bar{y}_i(x)) - \bar{y}_i(x) ,$$
# where $y$ is the measured emission sinogram, and $\bar{y}(x) = Ax + s$ the expectation of the measured data given the current
# estimate of the image $x$ and a linear (affine) forward model $A$ including the mean of known additive contaminations (randoms and scatter) $s$.
#

# %% [markdown]
# Exercise 1.1
# ------------
#
# Given the equations above, derive the update formula for the gradient of the Poisson log-likelihood (using sinogram data)
#
# (bonus question) How does the update formula change if we use listmode data instead of sinogram data?

# %% [markdown]
# Solution 1.1
# ------------
#
# In matrix notation, the gradient of the Poisson log-likelihood is given by:
# $$ \nabla_x \log L(y|x) = A^T \left( \frac{y}{\bar{y}(x)} - 1 \right) = A^T \left( \frac{y}{Ax + s} - 1 \right) .$$
#
# For a given image voxel $j$, the corresponding expression reads:
# $$ \frac{\partial \log L(y|x)} {\partial x_j} = \sum_{i=1}^m a_{ij} \left( \frac{y_i}{\sum_{k=1}^n a_{ik} x_k + s_i} - 1 \right) .$$
#
# Using a list of event "e" instead of a sinogram, the gradient of the Poisson log-likelihood becomes:
# $$ \frac{\partial \log L(y|x)} {\partial x_j} = \sum_{\text{events} \ e} a_{i_ej} \frac{1}{\sum_{k=1}^n a_{i_ek} x_k + s_{i_e}} -  \sum_{i=1}^m a_{ij} 1.$$
#
# **Note:**
# - SIRF (using the STIR backend) already provides implementations of the (TOF) PET acquisition forward model and
#   the gradient of the Poisson log-likelihood such that we do not have to re-implement these.
# - using SIRF with STIR, this gradient can be evaluated in listmode
# - if the number of listmode events is much smaller compared to the number of (TOF) sinogram bins, evaluating the gradient
#   in listmode can be more efficient.
