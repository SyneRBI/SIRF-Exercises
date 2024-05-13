# %% [markdown]
# Sinogram and Listmode OSEM using sirf.STIR
# ==========================================
#
# Using the theory learnings from the previous "theory" notebook, we will now learn how to perform
# PET reconstruction of emission data in listmode and sinogram format using (sinogram and listmode)
# objective function objects of the sirf.STIR library.
#
# We will see that standard OSEM reconstruction can be seen as a sequence of image update block,
# where the update in each block is related to the gradient of the Poisson loglikelihood objective function.
#
# Understanding these OSEM update blocks is the first key step for implementing a pytorch-based feed-forward
# neural network for PET image reconstruction also containing OSEM-like update blocks.

# %% [markdown]
# Import modules and define file names
# ------------------------------------

# %%
import sirf.STIR
import numpy as np
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
recon_output_file: str = str(output_path / "recon")
lm_recon_output_file: str = str(output_path / "lm_recon")
nxny: tuple[int, int] = (127, 127)
num_subsets: int = 21
num_iter: int = 1
num_scatter_iter: int = 3

# create the output directory
output_path.mkdir(exist_ok=True)

# engine's messages go to files, except error messages, which go to stdout
_ = sirf.STIR.MessageRedirector("info.txt", "warn.txt")

# %% [markdown]
# Read the listmode data and create a sinogram template
# -----------------------------------------------------

# %%
sirf.STIR.AcquisitionData.set_storage_scheme("memory")
listmode_data = sirf.STIR.ListmodeData(list_file)
acq_data_template = listmode_data.acquisition_data_template()
print(acq_data_template.get_info())

# %% [markdown]
# Conversion of listmode to sinogram data (needed for scatter estimation)
# -----------------------------------------------------------------------

# %%
# create listmode-to-sinograms converter object
lm2sino = sirf.STIR.ListmodeToSinograms()

# set input, output and template files
lm2sino.set_input(listmode_data)
lm2sino.set_output_prefix(emission_sinogram_output_prefix)
lm2sino.set_template(acq_data_template)

# get the start and end time of the listmode data
frame_start = float(
    [
        x
        for x in listmode_data.get_info().split("\n")
        if x.startswith("Time frame start")
    ][0]
    .split(": ")[1]
    .split("-")[0]
)
frame_end = float(
    [
        x
        for x in listmode_data.get_info().split("\n")
        if x.startswith("Time frame start")
    ][0]
    .split(": ")[1]
    .split("-")[1]
    .split("(")[0]
)
# set interval
lm2sino.set_time_interval(frame_start, frame_end)
# set up the converter
lm2sino.set_up()

# convert (need it for the scatter estimate)
lm2sino.process()
acq_data = lm2sino.get_output()

# %% [markdown]
# Estimation of random coincidences
# ---------------------------------

# %%
randoms_filepath = Path(f"{randoms_sinogram_output_prefix}.hs")

if not randoms_filepath.exists():
    print("estimting randoms")
    randoms = lm2sino.estimate_randoms()
    randoms.write(randoms_sinogram_output_prefix)
else:
    print("reading randoms from {randoms_filepath}")
    randoms = sirf.STIR.AcquisitionData(str(randoms_filepath))


# %% [markdown]
# Setup of the acquisition model
# ------------------------------

# %%
# select acquisition model that implements the geometric
# forward projection by a ray tracing matrix multiplication
acq_model = sirf.STIR.AcquisitionModelUsingRayTracingMatrix()
acq_model.set_num_tangential_LORs(1)


# %% [markdown]
# Calculation the attenuation sinogram
# ------------------------------------

# %%
# read attenuation image and display a single slice
attn_image = sirf.STIR.ImageData(attn_file)

# create attenuation factors
asm_attn = sirf.STIR.AcquisitionSensitivityModel(attn_image, acq_model)
# converting attenuation image into attenuation factors (one for every bin)
asm_attn.set_up(acq_data)

acf_filepath = Path(f"{attenuation_sinogram_output_prefix}.hs")

if not acf_filepath.exists():
    ac_factors = acq_data.get_uniform_copy(value=1)
    print("applying attenuation (please wait, may take a while)...")
    asm_attn.unnormalise(ac_factors)
    ac_factors.write(attenuation_sinogram_output_prefix)
else:
    print(f"reading attenuation factors from {acf_filepath}")
    ac_factors = sirf.STIR.AcquisitionData(str(acf_filepath))

asm_attn = sirf.STIR.AcquisitionSensitivityModel(ac_factors)

# %% [markdown]
# Creation of the normalization factors (sensitivity sinogram)
# ------------------------------------------------------------

# %%
# create acquisition sensitivity model from normalisation data
asm_norm = sirf.STIR.AcquisitionSensitivityModel(norm_file)

asm = sirf.STIR.AcquisitionSensitivityModel(asm_norm, asm_attn)
asm.set_up(acq_data)
acq_model.set_acquisition_sensitivity(asm)

# %% [markdown]
# Estimation of scattered coincidences
# ------------------------------------

# %%
scatter_filepath: Path = Path(f"{scatter_sinogram_output_prefix}_{num_scatter_iter}.hs")

if not scatter_filepath.exists():
    print("estimating scatter (this will take a while!)")
    scatter_estimator = sirf.STIR.ScatterEstimator()
    scatter_estimator.set_input(acq_data)
    scatter_estimator.set_attenuation_image(attn_image)
    scatter_estimator.set_randoms(randoms)
    scatter_estimator.set_asm(asm_norm)
    # invert attenuation factors to get the correction factors,
    # as this is unfortunately what a ScatterEstimator needs
    acf_factors = acq_data.get_uniform_copy()
    acf_factors.fill(1 / ac_factors.as_array())
    scatter_estimator.set_attenuation_correction_factors(acf_factors)
    scatter_estimator.set_output_prefix(scatter_sinogram_output_prefix)
    scatter_estimator.set_num_iterations(num_scatter_iter)
    scatter_estimator.set_up()
    scatter_estimator.process()
    scatter_estimate = scatter_estimator.get_output()
else:
    print(f"reading scatter from file {scatter_filepath}")
    scatter_estimate = sirf.STIR.AcquisitionData(str(scatter_filepath))

# chain attenuation and ECAT8 normalisation
acq_model.set_background_term(randoms + scatter_estimate)

# %% [markdown]
# Setup of the Poisson loglikelihood objective function ($logL(y,x)$) in sinogram mode
# ------------------------------------------------------------------------------------

# %%
initial_image = acq_data.create_uniform_image(value=1, xy=nxny)

# create objective function
obj_fun = sirf.STIR.make_Poisson_loglikelihood(acq_data)
obj_fun.set_acquisition_model(acq_model)
obj_fun.set_num_subsets(num_subsets)
obj_fun.set_up(initial_image)

# %% [markdown]
# Image reconstruction (optimization of the Poisson logL objective function) using sinogram OSEM
# ----------------------------------------------------------------------------------------------

# %%
if not Path(f"{recon_output_file}.hv").exists():
    reconstructor = sirf.STIR.OSMAPOSLReconstructor()
    reconstructor.set_objective_function(obj_fun)
    reconstructor.set_num_subsets(num_subsets)
    reconstructor.set_num_subiterations(num_iter * num_subsets)
    reconstructor.set_input(acq_data)
    reconstructor.set_up(initial_image)
    reconstructor.set_current_estimate(initial_image)
    reconstructor.process()
    ref_recon = reconstructor.get_output()
    ref_recon.write(recon_output_file)
else:
    ref_recon = sirf.STIR.ImageData(f"{recon_output_file}.hv")

vmax = np.percentile(ref_recon.as_array(), 99.999)

# %%
current_estimate = ref_recon.copy()
input_img = acq_data.create_uniform_image(value=1, xy=nxny)
np.random.seed(0)
input_img.fill(np.random.rand(*input_img.shape) * (obj_fun.get_subset_sensitivity(0).as_array() > 0) * current_estimate.max())

hess_out_img = obj_fun.accumulate_Hessian_times_input(current_estimate, input_img, subset=0)

# %%
# repeat the calculation using the LM objective function
# define objective function to be maximized as
# Poisson logarithmic likelihood (with linear model for mean)
lm_obj_fun = (
    sirf.STIR.PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin()
)
lm_obj_fun.set_acquisition_model(acq_model)
lm_obj_fun.set_acquisition_data(listmode_data)
lm_obj_fun.set_num_subsets(num_subsets)
lm_obj_fun.set_up(initial_image)

hess_out_img_lm = lm_obj_fun.accumulate_Hessian_times_input(current_estimate, input_img, subset=0)

# %%
# verify hessian calculation

acq_model.set_up(acq_data, initial_image)
acq_model.num_subsets = num_subsets
acq_model.subset_num = 0

# get the linear (Ax) part of the Ax + b affine acq. model
lin_acq_model = acq_model.get_linear_acquisition_model()
lin_acq_model.num_subsets = num_subsets
lin_acq_model.subset_num = 0

# for the Hessian "multiply" we need the linear part of the acquisition model applied to the input image
input_img_fwd = lin_acq_model.forward(input_img)
current_estimate_fwd = acq_model.forward(current_estimate)
h = -acq_model.backward(acq_data*input_img_fwd / (current_estimate_fwd*current_estimate_fwd))
h2 = -acq_model.backward(acq_data*input_img_fwd / (current_estimate_fwd*current_estimate_fwd + 1e-8))


# %%

fig, ax = plt.subplots(2, 6, figsize=(18, 6), tight_layout=True)
ax[0,0].imshow(current_estimate.as_array()[71, :, :], cmap = 'Greys')
ax[0,1].imshow(input_img.as_array()[71, :, :], cmap = 'Greys')
ax[0,2].imshow(hess_out_img.as_array()[71, :, :], cmap = 'Greys', vmin = -5000, vmax = -1000)
ax[0,3].imshow(hess_out_img_lm.as_array()[71, :, :], cmap = 'Greys', vmin = -5000, vmax = -1000)
ax[0,4].imshow(h.as_array()[71, :, :], cmap = 'Greys', vmin = -5000, vmax = -1000)
ax[0,5].imshow(h2.as_array()[71, :, :], cmap = 'Greys', vmin = -5000, vmax = -1000)
ax[1,2].imshow(hess_out_img.as_array()[71, :, :], cmap = 'Greys', vmin = -100000, vmax = hess_out_img.max())
ax[1,3].imshow(hess_out_img_lm.as_array()[71, :, :], cmap = 'Greys', vmin = -100000, vmax = hess_out_img.max())
ax[1,4].imshow(h.as_array()[71, :, :], cmap = 'Greys', vmin = -100000, vmax = hess_out_img.max())
ax[1,5].imshow(h2.as_array()[71, :, :], cmap = 'Greys', vmin = -100000, vmax = hess_out_img.max())
ax[0,0].set_title('current estimate', fontsize = 'medium')
ax[0,1].set_title('input', fontsize = 'medium')
ax[0,2].set_title('sino Hessian multiply', fontsize = 'medium')
ax[0,3].set_title('neg. LM Hessian multiply', fontsize = 'medium')
ax[0,4].set_title('manual Hessian multiply', fontsize = 'medium')
ax[0,5].set_title('manual Hessian multiply + eps', fontsize = 'medium')
ax[1,0].set_axis_off()
ax[1,1].set_axis_off()
fig.show()