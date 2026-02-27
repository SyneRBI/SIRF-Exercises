# STIR listmode (LM) Deep learning (DL) reconstruction notebooks

## Structure of the exercises

- Intro / motivation: `00_introduction.ipynb`
   - problem setting (DL recon network to maps from "low quality" to "higher quality" images)
   - Why listmode and not sinograms?

- Running sinogram and listmode OSEM reconstruction in sirf.STIR: `01_SIRF_listmode_recon.ipynb`
   - learn how to run (listmode) OSEM in sirf.STIR
   - understand the relation between the OSEM update and the gradient of the Poisson logL

- A deep dive into differenet (image) array classes (SIRF images vs numpy arrays vs torch tensors): `02_SIRF_vs_torch_arrays.ipynb`
   - differences between sirf.STIR.ImageData, numpy arrays and pytorch tensors
   - how to convert from SIRF images to torch tensors and back

- Defining custom (non-pytorch) layers that are compatible with pytorch's autograd functionality: `03_custom_torch_layers.ipynb`
   - basic of gradient backpropagation
   - understand what needs to be implemented in the backward pass based on a simply numpy matrix
     multiplication layer

- Defining a custom (listmode) Poisson logL gradient step layer using sirf.STIR and pytorch: `04_custom_sirf_Poisson_logL_layer.ipynb`
   - use of sirf.STIR for a step in the direction of the gradient of the Poisson logL
   - understanding of the respective Jacobian vector product for the backward pass
   - combining the Poisson logL gradient step layer into a OSEM update layer

- Demo training of a minimal unrolled variational network: `05_custrom_unrolled_varnet.ipynb`
   - combination of OSEM update layers and a CNN into an unrolled variational network
   - demo supervised training based on a single low count data set + high count reference image 

- Outlook: `06_outlook.ipynb`
