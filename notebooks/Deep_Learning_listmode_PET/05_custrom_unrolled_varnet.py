# %% [markdown]
# Learning objectives
# ===================
#
# 1. Exercise 1: Implement a custom layer that calculates the Poisson log-likelihood.
#                How to define the backward pass?
# 2. Exercise 2: Using the custom layer gradient logL layer, define EM step layer.

# %%
import sirf.STIR
import torch
import numpy as np
from pathlib import Path
from sirf.Utilities import examples_data_path
from scipy.ndimage import gaussian_filter

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


class SIRFPoissonlogLGradLayer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        objective_function,
        sirf_template_image: sirf.STIR.ImageData,
        subset: int,
    ) -> torch.Tensor:

        # we use the context object ctx to store the matrix and other variables that we need in the backward pass
        ctx.device = x.device
        ctx.objective_function = objective_function
        ctx.dtype = x.dtype
        ctx.subset = subset
        ctx.sirf_template_image = sirf_template_image

        # setup a new sirf.STIR ImageData object
        x_sirf = sirf_template_image.clone()
        # convert torch tensor to sirf image via numpy
        x_sirf.fill(x.cpu().numpy()[0, 0, ...])

        # save the input sirf.STIR ImageData for the backward pass
        ctx.x_sirf = x_sirf

        # calculate the gradient of the Poisson log likelihood using SIRF
        g_np = objective_function.gradient(x_sirf, subset).as_array()

        # convert back to torch tensor
        y = (
            torch.tensor(g_np, device=ctx.device, dtype=ctx.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        return y

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, None, None, None]:
        if grad_output is None:
            return None, None, None, None
        else:
            # convert torch tensor to sirf image via numpy
            ctx.sirf_template_image.fill(grad_output.cpu().numpy()[0, 0, ...])

            # calculate the Jacobian vector product (the Hessian applied to an image) using SIRF
            back_sirf = ctx.objective_function.accumulate_Hessian_times_input(
                ctx.x_sirf, ctx.sirf_template_image, ctx.subset
            )

            # convert back to torch tensor via numpy
            back = (
                torch.tensor(back_sirf.as_array(), device=ctx.device, dtype=ctx.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
            )

            return back, None, None, None


# %%
class SIRFOSEMStepLayer(torch.nn.Module):
    def __init__(
        self,
        objective_function,
        sirf_template_image: sirf.STIR.ImageData,
        subset: int,
        device: str,
    ) -> None:
        super().__init__()
        self._objective_function = objective_function
        self._sirf_template_image = sirf_template_image
        self._subset = subset

        self._poisson_logL_grad_layer = SIRFPoissonlogLGradLayer.apply

        self._inv_sens_image = 1.0 / torch.tensor(
            objective_function.get_subset_sensitivity(subset).as_array(),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0).unsqueeze(0)
        torch.nan_to_num(self._inv_sens_image, posinf=0, out=self._inv_sens_image)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grad_x = self._poisson_logL_grad_layer(
            x, self._objective_function, self._sirf_template_image, self._subset
        )

        return x + x * self._inv_sens_image * grad_x


# %%
class UnrolledOSEMVarNet(torch.nn.Module):
    def __init__(
        self,
        objective_function,
        sirf_template_image: sirf.STIR.ImageData,
        convnet: torch.nn.Module,
        device: str,
    ) -> None:
        super().__init__()
        self._osem_step_layer0 = SIRFOSEMStepLayer(
            objective_function, sirf_template_image, 0, device
        )
        self._osem_step_layer1 = SIRFOSEMStepLayer(
            objective_function, sirf_template_image, 1, device
        )
        self._osem_step_layer2 = SIRFOSEMStepLayer(
            objective_function, sirf_template_image, 2, device
        )
        self._osem_step_layer3 = SIRFOSEMStepLayer(
            objective_function, sirf_template_image, 3, device
        )
        self._convnet = convnet
        self._relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self._relu(self._convnet(x) + self._osem_step_layer0(x))
        x2 = self._relu(self._convnet(x1) + self._osem_step_layer1(x1))
        x3 = self._relu(self._convnet(x2) + self._osem_step_layer2(x2))
        x4 = self._relu(self._convnet(x3) + self._osem_step_layer2(x3))

        return x4


# %%

lm_ref_recon = sirf.STIR.ImageData(f"{lm_recon_output_file}.hv")
x_t = (
    torch.tensor(
        lm_ref_recon.as_array(), device=dev, dtype=torch.float32, requires_grad=False
    )
    .unsqueeze(0)
    .unsqueeze(0)
)

cnn = torch.nn.Sequential(
    torch.nn.Conv3d(1, 5, 5, padding="same", bias=False),
    torch.nn.Conv3d(5, 5, 5, padding="same", bias=False),
    torch.nn.ReLU(),
    torch.nn.Conv3d(5, 5, 5, padding="same", bias=False),
    torch.nn.Conv3d(5, 5, 5, padding="same", bias=False),
    torch.nn.ReLU(),
    torch.nn.Conv3d(5, 1, 5, padding="same", bias=False),
).to(dev)


varnet = UnrolledOSEMVarNet(lm_obj_fun, initial_image, cnn, dev)

# define a high quality traget image
target = (
    torch.tensor(
        gaussian_filter(x_t.cpu().numpy()[0, 0, ...], 0.7),
        dtype=torch.float32,
        device=dev,
    )
    .unsqueeze(0)
    .unsqueeze(0)
)

optimizer = torch.optim.Adam(varnet._convnet.parameters(), lr=1e-3)
# define the loss function
loss_fct = torch.nn.MSELoss()

# %%
for i in range(1):
    # pass the input mini-batch through the network
    prediction = varnet(x_t)
    # calculate the MSE loss between the prediction and the target
    loss = loss_fct(prediction, target)
    # backpropagate the gradient of the loss through the network
    # (needed to update the trainable parameters of the network with an optimizer)
    optimizer.zero_grad()
    loss.backward()
    # update the trainable parameters of the network with the optimizer
    optimizer.step()
    print(i, loss.item())
