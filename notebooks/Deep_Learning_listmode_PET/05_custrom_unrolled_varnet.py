# %% [markdown]
# Creating a custom unrolled variational network for listmode PET data
# ====================================================================
#
# Learning objectives
# -------------------
#
# 1. Learn how to implement and train a custom unrolled variational network fusing updates
#    from listmode OSEM blocks and CNN blocks

# %%
import sirf.STIR
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sirf.Utilities import examples_data_path

# acq_time must be 1min
acq_time: str = "1min"

data_path: Path = Path(examples_data_path("PET")) / "mMR"
list_file: str = str(data_path / "list.l.hdr")
norm_file: str = str(data_path / "norm.n.hdr")
attn_file: str = str(data_path / "mu_map.hv")

output_path: Path = Path(f"recons_{acq_time}")
emission_sinogram_output_prefix: str = str(output_path / "emission_sinogram")
scatter_sinogram_output_prefix: str = str(output_path / "scatter_sinogram")
randoms_sinogram_output_prefix: str = str(output_path / "randoms_sinogram")
attenuation_sinogram_output_prefix: str = str(output_path / "acf_sinogram")

num_scatter_iter: int = 3

lm_recon_output_file: str = str(output_path / "lm_recon")
lm_60min_recon_output_file: str = str(Path(f"recons_60min") / "lm_recon")
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
lm_obj_fun.set_cache_max_size(1000000000)
lm_obj_fun.set_cache_path(str(output_path))
print("setting up listmode objective function ...")
lm_obj_fun.set_up(initial_image)

# %% [markdown]
# Setup of OSEM update layer
# --------------------------
#
# See notebook 04.

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
            # convert torch tensor to sirf image via numpy
            ctx.sirf_template_image.fill(grad_output.cpu().numpy()[0, 0, ...])

            # calculate the Jacobian vector product (the Hessian applied to an image) using SIRF
            back_sirf = ctx.objective_function.multiply_with_Hessian(
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
        grad_x: torch.Tensor = self._poisson_logL_grad_layer(
            x, self._objective_function, self._sirf_template_image, self._subset
        )
        return x + x * self._inv_sens_image * grad_x


# %% [markdown]
# Exercise 5.1
# ------------
#
# Implement the forward pass of the unrolled OSEM Variational Network with 2 blocks shown below.
# Start from the code below and fill in the missing parts.

# %%
class UnrolledOSEMVarNet(torch.nn.Module):
    def __init__(
        self,
        objective_function,
        sirf_template_image: sirf.STIR.ImageData,
        convnet: torch.nn.Module,
        device: str,
    ) -> None:
        """Unrolled OSEM Variational Network with 2 blocks

        Parameters
        ----------
        objective_function : sirf.STIR objetive function
            (listmode) Poisson logL objective function
            that we use for the OSEM updates
        sirf_template_image : sirf.STIR.ImageData
            used for the conversion between torch tensors and sirf images
        convnet : torch.nn.Module
            a (convolutional) neural network that maps a minibatch tensor 
            of shape [1,1,spatial_dimensions] onto a minibatch tensor of the same shape
        device : str
            device used for the calculations
        """
        super().__init__()

        # OSEM update layer using the 1st subset of the listmode data
        self._osem_step_layer0 = OSEMUpdateLayer(
            objective_function, sirf_template_image, 0, device
        )

        # OSEM update layer using the 2nd subset of the listmode data
        self._osem_step_layer1 = OSEMUpdateLayer(
            objective_function, sirf_template_image, 1, device
        )
        self._convnet = convnet
        self._relu = torch.nn.ReLU()

        # trainable parameters for the fusion of the OSEM update and the CNN output in the two blocks
        self._fusion_weight0 = torch.nn.Parameter(
            torch.ones(1, device=device, dtype=torch.float32)
        )
        self._fusion_weight1 = torch.nn.Parameter(
            torch.ones(1, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of the Unrolled OSEM Variational Network

        Parameters
        ----------
        x : torch.Tensor
            minibatch tensor of shape [1,1,spatial_dimensions] containing the image

        Returns
        -------
        torch.Tensor
            minibatch tensor of shape [1,1,spatial_dimensions] containing the output of the network
        """

        # =============================================================
        # =============================================================
        # YOUR CODE HERE
        #
        # forward pass contains two blocks where each block
        # consists of a fusion of the OSEM update and the CNN output
        #
        # the fusion is a weighted sum of the OSEM update and the CNN output
        # using the respective fusion weights
        #
        # =============================================================
        # =============================================================


# %%
# load the reference OSEM reconstruction that we use a input our network
lm_ref_recon = sirf.STIR.ImageData(f"{lm_recon_output_file}.hv")
x_t = (
    torch.tensor(
        lm_ref_recon.as_array(), device=dev, dtype=torch.float32, requires_grad=False
    )
    .unsqueeze(0)
    .unsqueeze(0)
)

# define a minimal CNN that we use within the unrolled OSEM Variational Network
cnn = torch.nn.Sequential(
    torch.nn.Conv3d(1, 5, 5, padding="same", bias=False),
    torch.nn.Conv3d(5, 5, 5, padding="same", bias=False),
    torch.nn.PReLU(device=dev),
    torch.nn.Conv3d(5, 5, 5, padding="same", bias=False),
    torch.nn.Conv3d(5, 5, 5, padding="same", bias=False),
    torch.nn.PReLU(device=dev),
    torch.nn.Conv3d(5, 1, 1, padding="same", bias=False),
).to(dev)


# setup the unrolled OSEM Variational Network using the sirf.STIR listmode objective function
# and the CNN
varnet = UnrolledOSEMVarNet(lm_obj_fun, initial_image, cnn, dev)

# %% [markdown]
#
# Supervised optimization the network parameters
# ----------------------------------------------
#
# The following cells demonstrate how to optimize the network parameters
# using a high quality target image (supervised learning).
# Here, we use the reconstruction of the 60min listmode data as the target image.

# %%
# define the high quality target image (mini-batch)
lm_60min_ref_recon = sirf.STIR.ImageData(f"{lm_60min_recon_output_file}.hv")

# we have to scale the 60min reconstruction, since it is not reconcstructed in kBq/ml
scale_factor = lm_ref_recon.as_array().mean() / lm_60min_ref_recon.as_array().mean()
lm_60min_ref_recon *= scale_factor

target = (
    torch.tensor(
        lm_60min_ref_recon.as_array(), device=dev, dtype=torch.float32, requires_grad=False
    )
    .unsqueeze(0)
    .unsqueeze(0)
)

# %% [markdown]
# To train the network weights, we need to define an optimizer and a loss function.
# Here we use the Adam optimizer with a learning rate of 1e-3 and the Mean Squared Error (MSE) loss function.



# %%
optimizer = torch.optim.Adam(varnet._convnet.parameters(), lr=1e-3)
# define the loss function
loss_fct = torch.nn.MSELoss()

# %%
# run 10 updates of the model parameters using backpropagation of the
# gradient of the loss function and the Adam optimizer

num_epochs = 50
training_loss = torch.zeros(num_epochs)

for i in range(num_epochs):
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
    # save the training loss
    training_loss[i] = loss.item()

# %%
# visualize the results
vmax = float(target.max())
sl = 71

fig1, ax1 = plt.subplots(2, 3, figsize=(9, 6), tight_layout=True)
ax1[0, 0].imshow(x_t.cpu().numpy()[0, 0, sl, :, :], cmap="Greys", vmin=0, vmax=vmax)
ax1[0, 1].imshow(
    prediction.detach().cpu().numpy()[0, 0, sl, :, :], cmap="Greys", vmin=0, vmax=vmax
)
ax1[0, 2].imshow(target.cpu().numpy()[0, 0, sl, :, :], cmap="Greys", vmin=0, vmax=vmax)
ax1[1, 0].imshow(
    x_t.cpu().numpy()[0, 0, sl, :, :] - target.cpu().numpy()[0, 0, sl, :, :],
    cmap="seismic",
    vmin=-0.01,
    vmax=0.01,
)
ax1[1, 1].imshow(
    prediction.detach().cpu().numpy()[0, 0, sl, :, :]
    - target.cpu().numpy()[0, 0, sl, :, :],
    cmap="seismic",
    vmin=-0.01,
    vmax=0.01,
)

ax1[0, 0].set_title("network input")
ax1[0, 1].set_title("network output")
ax1[0, 2].set_title("target")
ax1[1, 0].set_title("network input - target")
ax1[1, 1].set_title("network output - target")
fig1.show()

# plot the training loss
fig2, ax2 = plt.subplots()
ax2.plot(training_loss.cpu().numpy())
ax2.set_xlabel("epoch")
ax2.set_ylabel("training loss")
fig2.show()
