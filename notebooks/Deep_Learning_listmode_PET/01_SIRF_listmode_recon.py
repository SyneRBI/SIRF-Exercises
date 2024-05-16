# %% [markdown]
# Sinogram and Listmode OSEM using sirf.STIR
# ==========================================
#
# Using the learnings from the previous "theory" notebook, we will now learn how to perform
# PET reconstruction of emission data in listmode and sinogram format using (sinogram and listmode)
# objective function objects of the sirf.STIR library.
#
# We will see that standard OSEM reconstruction can be seen as a sequence of image update "blocks",
# where the update in each block is related to the gradient of the Poisson loglikelihood objective function.
#
# Understanding these OSEM update blocks is the first key step for implementing a pytorch-based feed-forward
# neural network for PET image reconstruction also containing OSEM-like update blocks.
#
# Learning objectives of this notebook
# ------------------------------------
# 1. Understanding how to setup a Poisson loglikelihood objective functions in sinogram and listmode mode.
# 2. Understanding how to perform sinogram / listmode OSEM reconstruction using sirf.STIR high-level API.
# 3. Implementing a simple DIY OSEM reconstruction using the gradient of the Poisson loglikelihood.

# %% [markdown]
# Import modules
# --------------

# %%
import sirf.STIR
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
from sirf.Utilities import examples_data_path

# %% [markdown]
# Download the 60min mMR NEMA data, if not present
# ------------------------------------------------

# %% 
if not (Path("..") / ".." / "data" / "PET" / "mMR" / "NEMA_IQ" / "20170809_NEMA_60min_UCL.l.hdr").exists():
    retval = subprocess.call("../../scripts/download_PET_data.sh", shell=True)

# %% [markdown]
# Define variables and file names
# -------------------------------

# %%
# we have a 1min and 60min acquisition of the NEMA IQ phantom acquired on a Siemens mMR
# choose the acquisition time "1min" or "60min" - start with "1min"
acq_time: str = "1min"

data_path: Path = Path(examples_data_path("PET")) / "mMR"

if acq_time == "1min":
    list_file: str = str(data_path / "list.l.hdr")
elif acq_time == "60min":
    # you need to run the "download_data.sh" script to get the data of the long 60min acq.
    list_file: str = str(Path("..") / ".." / "data" / "PET" / "mMR" / "NEMA_IQ" / "20170809_NEMA_60min_UCL.l.hdr")
else:
    raise ValueError("Please choose acq_time to be either '1min' or '60min'")

attn_file: str = str(data_path / "mu_map.hv")
norm_file: str = str(data_path / "norm.n.hdr")
output_path: Path = Path(f"recons_{acq_time}")
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
# acq_model.set_num_tangential_LORs(10)
acq_model.set_num_tangential_LORs(1)

# %% [markdown]
# Calculation of the attenuation sinogram
# ---------------------------------------

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

# add scatter plus randoms estimated to the background term of the acquisition model
acq_model.set_background_term(randoms + scatter_estimate)

# %% [markdown]
# Setup of the Poisson loglikelihood objective function in sinogram mode
# ----------------------------------------------------------------------

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

fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
ax.imshow(ref_recon.as_array()[71, :, :], cmap="Greys", vmin=0, vmax=vmax)
fig.show()

# %% [markdown]
# Exercise 1.1
# ------------
#
# Perform the gradient ascent step
# $$ x^+ = x + \alpha \nabla_x logL(y,x) $$
# on the initial image x using a constant scalar step size $\alpha=0.001$ by calling
# the `gradient()` method of the objective function.
# Use the first (0th) subset of the data for the gradient calculation.

# %%
#
# ==============
# YOUR CODE HERE
# ==============
#

# %%
# to view the solution, execute the this cell
# %load snippets/solution_1_1.py

# %% [markdown]
# Exercise 1.2
# ------------
#
# Given the fact that the OSEM update can be written as
# $$ x^+ = x + t \nabla_x logL(y,x) $$
# with the non-scalar step size
# $$ t = \frac{x}{s} $$
# where $s$ is the (subset) "sensitivity image", perform an OSEM update on the initial image
# by using the `get_subset_sensitivity()` method of the objective function and the first subset.
# Print the maximum value of the updated image. What do you observe?

# %%
#
# ==============
# YOUR CODE HERE
# ==============
#

# %%
# to view the solution, execute the this cell
# %load snippets/solution_1_2.py

# %% [markdown]
# Exercise 1.3
# ------------
#
# Implement your own OSEM reconstruction by looping over the subsets and performing the
# OSEM update for each subset.

# %%
# initialize the reconstruction with ones where the sensitivity image is greater than 0
# all other values are set to zero and are not updated during reconstruction
recon = initial_image.copy()
recon.fill(obj_fun.get_subset_sensitivity(0).as_array() > 0)
#
# ==============
# YOUR CODE HERE
# ==============
#

# %%
# to view the solution, execute the this cell
# %load snippets/solution_1_3.py


# %% [markdown]
# Setup of the Poisson loglikelihood objective function logL(y,x) in listmode
# ---------------------------------------------------------------------------

# %%
# define the listmode objective function
lm_obj_fun = (
    sirf.STIR.PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin()
)
lm_obj_fun.set_acquisition_model(acq_model)
lm_obj_fun.set_acquisition_data(listmode_data)
lm_obj_fun.set_num_subsets(num_subsets)
lm_obj_fun.set_cache_max_size(1000000000)
lm_obj_fun.set_cache_path(str(output_path))

# %% [markdown]
# Reconstruction (optimization of the Poisson logL objective function) using listmode OSEM
# ----------------------------------------------------------------------------------------

# %%
if not Path(f"{lm_recon_output_file}.hv").exists():
    lm_reconstructor = sirf.STIR.OSMAPOSLReconstructor()
    lm_reconstructor.set_objective_function(lm_obj_fun)
    lm_reconstructor.set_num_subsets(num_subsets)
    lm_reconstructor.set_num_subiterations(num_iter * num_subsets)
    lm_reconstructor.set_up(initial_image)
    lm_reconstructor.set_current_estimate(initial_image)
    lm_reconstructor.process()
    lm_ref_recon = lm_reconstructor.get_output()
    lm_ref_recon.write(lm_recon_output_file)
else:
    lm_ref_recon = sirf.STIR.ImageData(f"{lm_recon_output_file}.hv")

fig3, ax3 = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
ax3.imshow(lm_ref_recon.as_array()[71, :, :], cmap="Greys", vmin=0, vmax=vmax)
fig3.show()

# %% [markdown]
# Exercise 1.4
# ------------
# Repeat exercise 1.3 (OSEM reconstruction) using the listmode objective function to
# learn how to do a listmode OSEM update step.

# %%
#
# ==============
# YOUR CODE HERE
# ==============

# %%
# to view the solution, execute the cell below
# %load snippets/solution_1_4.py

# %% [markdown]
# Exercise 1.5
# ------------
# Rerun the sinogram and listmode reconstruction (first cells of the notebook)
# using the 60min acquisition data by adapting the `acq_time` variable.
# Make sure that you restart the kernel before running the cells and to rerun
# the all cells (including scatter and random estimation).
# We wil use the 60min reconstruction in our last notebook.
