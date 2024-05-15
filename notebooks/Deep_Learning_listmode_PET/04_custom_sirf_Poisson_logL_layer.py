# %% [markdown]
# Creating custom Poisson log likelihood gradient step and OSEM update layers
# ===========================================================================
#
# Learning objectives
# -------------------
#
# 1. Implement the forward and backward pass of a custom (pytorch autograd compatible) layer that
#    calculates the gradient Poisson log-likelihood.
# 2. Understand how to test whether the (backward pass) of the custom layer is implemented correctly,
#    such that gradient backpropagation works as expected.

# %%
import sirf.STIR
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sirf.Utilities import examples_data_path

data_path: Path = Path(examples_data_path("PET")) / "mMR"
output_path: Path = Path("recons")
list_file: str = str(data_path / "list.l.hdr")
norm_file: str = str(data_path / "norm.n.hdr")
attn_file: str = str(data_path / "mu_map.hv")
emission_sinogram_output_prefix: str = str(output_path / "emission_sinogram")
scatter_sinogram_output_prefix: str = str(output_path / "scatter_sinogram")
randoms_sinogram_output_prefix: str = str(output_path / "randoms_sinogram")
attenuation_sinogram_output_prefix: str = str(output_path / "acf_sinogram")
num_scatter_iter: int = 3

lm_recon_output_file: str = str(output_path / "lm_recon")
nxny: tuple[int, int] = (127, 127)
num_subsets: int = 21

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

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

acq_data = sirf.STIR.AcquisitionData(
    str(Path(f"{emission_sinogram_output_prefix}_f1g1d0b0.hs"))
)

# select acquisition model that implements the geometric
# forward projection by a ray tracing matrix multiplication
acq_model = sirf.STIR.AcquisitionModelUsingRayTracingMatrix()
acq_model.set_num_tangential_LORs(1)

randoms = sirf.STIR.AcquisitionData(str(Path(f"{randoms_sinogram_output_prefix}.hs")))

ac_factors = sirf.STIR.AcquisitionData(
    str(Path(f"{attenuation_sinogram_output_prefix}.hs"))
)
asm_attn = sirf.STIR.AcquisitionSensitivityModel(ac_factors)

asm_norm = sirf.STIR.AcquisitionSensitivityModel(norm_file)
asm = sirf.STIR.AcquisitionSensitivityModel(asm_norm, asm_attn)

asm.set_up(acq_data)
acq_model.set_acquisition_sensitivity(asm)

scatter_estimate = sirf.STIR.AcquisitionData(
    str(Path(f"{scatter_sinogram_output_prefix}_{num_scatter_iter}.hs"))
)
acq_model.set_background_term(randoms + scatter_estimate)

# setup an initial (template) image based on the acquisition data template
initial_image = acq_data_template.create_uniform_image(value=1, xy=nxny)

# load the reconstructed image from notebook 01
lm_ref_recon = sirf.STIR.ImageData(f"{lm_recon_output_file}.hv")

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
print("setting up listmode objective function ...")
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
# For the implementation we subclass `torch.autograd.Function` and implement the `forward()` and
# `backward()` methods.

# %% [markdown]
# Exercise 4.1
# ------------
#
# Using your knowledge of the Poisson log likelihood gradient (exercise 0.1) and the content of the notebook 03
# on custom layers, implement the forward and backward pass of a custom layer that calculates the gradient of the
# Poisson log likelihood using a SIRF objective function as shown in the figure below.
#
# # ![](figs/poisson_logL_grad_layer.drawio.svg)
#
# The next cell contains the skeleton of the custom layer. You need to fill in the missing parts in the forward and
# backward pass.

# %%
class SIRFPoissonlogLGradLayer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        objective_function,
        sirf_template_image: sirf.STIR.ImageData,
        subset: int,
    ) -> torch.Tensor:
        """(listmode) Poisson loglikelihood gradient layer forward pass

        Parameters
        ----------
        ctx : context object
            used to store objects that we need in the backward pass
        x : torch.Tensor
            minibatch tensor of shape [1,1,spatial_dimensions] containing the image
        objective_function : sirf (listmode) objective function
            the objective function that we use to calculate the gradient
        sirf_template_image : sirf.STIR.ImageData
            image template that we use to convert between torch tensors and sirf images
        subset : int
            subset number used for the gradient calculation

        Returns
        -------
        torch.Tensor
            minibatch tensor of shape [1,1,spatial_dimensions] containing the image
            containing the gradient of the (listmode) Poisson log likelihood at x
        """
        # we use the context object ctx to store objects that we need in the backward pass
        ctx.device = x.device
        ctx.objective_function = objective_function
        ctx.dtype = x.dtype
        ctx.subset = subset
        ctx.sirf_template_image = sirf_template_image

        # ==============================================================
        # ==============================================================
        # YOUR CODE HERE
        # ==============================================================
        # ==============================================================

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, None, None, None]:
        """(listmode) Poisson loglikelihood gradient layer backward pass

        Parameters
        ----------
        ctx : context object
            used to store objects that we need in the backward pass
        grad_output : torch.Tensor | None
            minibatch tensor of shape [1,1,spatial_dimensions] containing the gradient (called v in the autograd tutorial)
            https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#optional-reading-vector-calculus-using-autograd

        Returns
        -------
        tuple[torch.Tensor | None, None, None, None]
            the Jacobian-vector product of the Poisson log likelihood gradient layer
        """

        if grad_output is None:
            return None, None, None, None
        else:
            ctx.sirf_template_image.fill(grad_output.cpu().numpy()[0, 0, ...])

            # ==============================================================
            # ==============================================================
            # YOUR CODE HERE
            # --------------
            #
            # calculate the Jacobian-vector product of the Poisson log likelihood gradient layer
            # Hints: (1) try to derive the Jacobian of the gradient of the Poisson log likelihood gradient first
            #        (2) the sirf.STIR objective function has a method called `multiply_with_Hessian`
            #
            # ==============================================================
            # ==============================================================


# %% [markdown]
# To view the solution to the exercise, execute the next cell.

# %%
# %load snippets/solution_4_1.py

# %%
# convert to torch tensor and add the minibatch and channel dimensions
x_t = (
    torch.tensor(
        lm_ref_recon.as_array(), device=dev, dtype=torch.float32, requires_grad=False
    )
    .unsqueeze(0)
    .unsqueeze(0)
)

# setup our custom Poisson log likelihood gradient layer
poisson_logL_grad_layer = SIRFPoissonlogLGradLayer.apply
# perform the forward pass (calcuate the gradient of the Poisson log likelihood at x_t)
grad_x = poisson_logL_grad_layer(x_t, lm_obj_fun, initial_image, 0)


# %% [markdown]
# Implementing a OSEM update layer using our custom Poisson log likelihood gradient layer
# =======================================================================================
#
# Using our custom Poisson log likelihood gradient layer, we can now implement a custom OSEM update layer.
# Note that the OSEM update can be decomposed into a simple feedforward network consisting of basic arithmetic
# operations that are implemented in pytorch (pointwise multiplication and addition) as shown in the figure below.
#
# # ![](figs/osem_layer.drawio.svg)

# %% [markdown]
# Exercise 4.2
# ------------
# Implement the forward pass of a OSEM update layer using the Poisson log likelihood gradient layer that we implemented
# above.

# %%
class OSEMUpdateLayer(torch.nn.Module):
    def __init__(
        self,
        objective_function,
        sirf_template_image: sirf.STIR.ImageData,
        subset: int,
        device: str,
    ) -> None:
        """OSEM update layer

        Parameters
        ----------
        objective_function : sirf (listmode) objective function
            the objective function that we use to calculate the gradient
        sirf_template_image : sirf.STIR.ImageData
            image template that we use to convert between torch tensors and sirf images
        subset : int
            subset number used for the gradient calculation
        device : str
            device used for the calculations

        Returns
        -------
        torch.Tensor
            minibatch tensor of shape [1,1,spatial_dimensions] containing the OSEM
            update of the input image using the Poisson log likelihood objective function
        """
        super().__init__()
        self._objective_function = objective_function
        self._sirf_template_image: sirf.STIR.ImageData = sirf_template_image
        self._subset: int = subset

        self._poisson_logL_grad_layer = SIRFPoissonlogLGradLayer.apply

        # setup a tensor containng the inverse of the subset sensitivity image adding the minibatch and channel dimensions
        self._inv_sens_image: torch.Tensor = 1.0 / torch.tensor(
            objective_function.get_subset_sensitivity(subset).as_array(),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0).unsqueeze(0)
        # replace positive infinity values with 0 (voxels with 0 sensitivity)
        torch.nan_to_num(self._inv_sens_image, posinf=0, out=self._inv_sens_image)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of the OSEM update layer

        Parameters
        ----------
        x : torch.Tensor
            minibatch tensor of shape [1,1,spatial_dimensions] containing the image

        Returns
        -------
        torch.Tensor
            OSEM update image
        """

        # =======================================================================
        # =======================================================================
        # YOUR CODE HERE
        # USE ONLY BASIC ARITHMETIC OPERATIONS BETWEEN TORCH TENSORS!
        # =======================================================================
        # =======================================================================


# %% [markdown]
# To view the solution to the exercise, execute the next cell.

# %%
# %load snippets/solution_4_2.py

# %%
# define the OSEM update layer for subset 0
osem_layer0 = OSEMUpdateLayer(lm_obj_fun, initial_image, 0, dev)
# perform the forward pass
osem_updated_x_t = osem_layer0(x_t)

# %%

# show the input and output of the OSEM update layer
fig, ax = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
ax[0].imshow(x_t.cpu().numpy()[0, 0, 71, ...], cmap="Greys")
ax[1].imshow(osem_updated_x_t.cpu().numpy()[0, 0, 71, ...], cmap="Greys")
ax[2].imshow(
    osem_updated_x_t.cpu().numpy()[0, 0, 71, ...] - x_t.cpu().numpy()[0, 0, 71, ...],
    cmap="seismic",
    vmin=-0.01,
    vmax=0.01,
)
ax[0].set_title("input image")
ax[1].set_title("OSEM updated image")
ax[2].set_title("diffence image")
fig.show()
