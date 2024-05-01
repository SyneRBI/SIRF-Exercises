# STIR listmode (LM) Deep learning (DL) reconstruction notebooks

## Notebooks

0. Intro / motivation
   -> problem setting (DL recon network to maps from "low quality" to "higher quality" images)
   -> Why listmode and not sinograms?

1. A deep dive into differenet (image) array classes (SIRF images vs numpy arrays vs torch tensors)
   -> differences between the array classes
   -> how to convert from SIRF images to torch tensors and back
   -> 3D images vs 5D mini batches of 3D images

2. Defining a custom LM Poisson log likelihood gradient step layer
   -> how to use gradient and "hessian_applied" to define a custom layer
      that calculates a gradient ascent step w.r.t. to Poisson logL in LM
   -> how to implement backprop? why hessian applied?

3. Defining a custom  LM Poisson log likelihood EM network
   -> use layer of 3 to setup torch.nn.Module that does LM-MLEM using layers + simple arith. of 2.

4. Defining a network that combines LM EM steps and a learned regularizer
   -> combine network of 3. with learned regularizer

5. How to train on "real" data?
   -> data loader
   -> current timings and bottle necks

