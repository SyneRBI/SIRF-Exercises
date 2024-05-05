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
# The latter can be represented by a neural network containing learnable parameters which are optimized during training.
