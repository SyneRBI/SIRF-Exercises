# Contents
This directory contains a few basic notebooks to get you going. It is recommended to start with these
before tackling any of the others.

## Basic concepts of SIRF via Python
The [introduction](introduction.ipynb) notebook serves as a starting point for all SIRF jupyter notebooks. 
The notebook shows how MR, PET and CT images can be created and manipulated.

The [acquisition_model_mr_pet_ct](acquisition_model_mr_pet_ct.ipynb) notebook shows how to create a MR, PET and CT acquisition model and go from images to raw data and back again for each modality. (Do check notebooks for each modality for more information on the acquisition models.)

The [gradient_descent_mr_pet_ct notebook](gradient_descent_mr_pet_ct.ipynb) shows how to write
a simple gradient descent algorithm for MR, PET and CT (using CIL for the latter).

## Learning objectives

- Ensure you have a working version of SIRF (whether you're viewing this via a virtual machine, docker image, an Azure instance or a SIRF installation on your own machine).
By the end of these notebooks, you should feel more comfortable with:
- Get a feel for Jupyter notebooks (and the basic python they require)
- Reading and displaying images
- Doing basic simulations and reconstruction in three different modalities
- Querying for help to improve knowledge of SIRF functionality.