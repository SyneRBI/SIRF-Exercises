In matrix notation, the gradient of the Poisson log-likelihood is given by:
$$ \nabla_x \log L(y|x) = A^T \left( \frac{y}{\bar{y}(x)} - 1 \right) = A^T \left( \frac{y}{Ax + s} - 1 \right) .$$

For a given image voxel $j$, the corresponding expression reads:
$$ \frac{\partial \log L(y|x)} {\partial x_j} = \sum_{i=1}^m a_{ij} \left( \frac{y_i}{\sum_{k=1}^n a_{ik} x_k + s_i} - 1 \right) .$$

Using a list of detected coincidence events $e$ instead of a sinogram, the gradient of the Poisson log-likelihood becomes:
$$ \frac{\partial \log L(y|x)} {\partial x_j} = \sum_{\text{events} \ e} a_{i_ej} \frac{1}{\sum_{k=1}^n a_{i_ek} x_k + s_{i_e}} -  \sum_{i=1}^m a_{ij} 1, $$
where $i_e$ is the (TOF) sinogram bin corresponding to event $e$.

**Note:**
- SIRF (using the STIR backend) already provides implementations of the (TOF) PET acquisition forward model and
  the gradient of the Poisson log-likelihood such that we do not have to re-implement these.
- using SIRF with STIR, this gradient can be evaluated in listmode
- if the number of listmode events is much smaller compared to the number of (TOF) sinogram bins, evaluating the gradient
  in listmode can be more efficient.

