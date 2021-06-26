# Contents

Jupyter notebooks for the PET exercises. Recommended order:

## Week 1
1. First look at the [Introductory/ notebooks](../Introductory/).
2. [display\_and\_projection](display_and_projection.ipynb) repeats some of that material but goes in a bit more detail on `PETAcquisitionData`.
2. [image\_creation\_and\_simulation](image_creation_and_simulation.ipynb) shows how to create some images with geometric shapes, and explains attenuation modelling etc.
3. [OSEM\_reconstruction](OSEM_reconstruction.ipynb) shows how to run a `sirf.STIR` class for OSEM reconstruction.
4. [reconstruct\_measured\_data](reconstruct_measured_data.ipynb) goes into detail on how to reconstruct data from the Siemens mMR.

## Week 2
1. [ML\_reconstruction](ML_reconstruction.ipynb) illustrates Maximum Likelihood Estimation via OSEM and gradient ascent.
2. [DIY_OSEM](DIY_OSEM.ipynb) invites you to write MLEM and OSEM yourself using SIRF functionality (optional).
3. [MAPEM](MAPEM.ipynb) is an exercise to implement the MAP-EM algorithm for the regularised objective function where a quadratic prior is added to the Poisson log-likelihood.
