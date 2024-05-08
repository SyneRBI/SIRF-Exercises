# %%
import sirf.STIR
import matplotlib.pyplot as plt
from pathlib import Path
from sirf.Utilities import examples_data_path

data_path: Path = Path(examples_data_path("PET")) / "mMR"
list_file: str = str(data_path / "list.l.hdr")
norm_file: str = str(data_path / "norm.n.hdr")
attn_file: str = str(data_path / "mu_map.hv")
output_file: str = str(data_path / "recon")
emission_sinogram_output_prefix: str = str(Path("lm_recons") / "emission_sinogram")
scatter_sinogram_output_prefix: str = str(Path("lm_recons") / "scatter_sinogram")
randoms_sinogram_output_prefix: str = str(Path("lm_recons") / "randoms_sinogram")
attenuation_sinogram_output_prefix: str = str(Path("lm_recons") / "acf_sinogram")
nxny: tuple[int, int] = (127, 127)
input_interval: tuple[int, int] = (0, 50)
num_subsets: int = 21
num_iter: int = 2
num_scatter_iter: int = 3
storage: str = "file"
use_gpu: bool = True


# %%
## engine's messages go to files, except error messages, which go to stdout
# _ = sirf.STIR.MessageRedirector('info.txt', 'warn.txt')

# select acquisition data storage scheme
sirf.STIR.AcquisitionData.set_storage_scheme(storage)
listmode_data = sirf.STIR.ListmodeData(list_file)
acq_data_template = listmode_data.acquisition_data_template()
print(acq_data_template.get_info())

# %%
# listmode to sinogram conversion
# -------------------------------

# create listmode-to-sinograms converter object
lm2sino = sirf.STIR.ListmodeToSinograms()

# set input, output and template files
lm2sino.set_input(listmode_data)
lm2sino.set_output_prefix(emission_sinogram_output_prefix)
lm2sino.set_template(acq_data_template)

# set interval
lm2sino.set_time_interval(input_interval[0], input_interval[1])
# set up the converter
lm2sino.set_up()

# convert (need it for the scatter estimate)
lm2sino.process()
acq_data = lm2sino.get_output()

# %%
# randoms estimation
# ------------------

randoms_filepath = Path(f"{randoms_sinogram_output_prefix}.hs")

if not randoms_filepath.exists():
    print("estimting randoms")
    randoms = lm2sino.estimate_randoms()
    randoms.write(randoms_sinogram_output_prefix)
else:
    print("reading randoms from {randoms_filepath}")
    randoms = sirf.STIR.AcquisitionData(str(randoms_filepath))


# %%
# setup of attenuation and normalisation factors
# ----------------------------------------------

# select acquisition model that implements the geometric
# forward projection by a ray tracing matrix multiplication
acq_model = sirf.STIR.AcquisitionModelUsingRayTracingMatrix()
# acq_model.set_num_tangential_LORs(10)
acq_model.set_num_tangential_LORs(1)

# %% 
# calcuate the attenuation sinogram
# ---------------------------------

# read attenuation image and display a single slice
attn_image = sirf.STIR.ImageData(attn_file)

# show the attenuation image
fig, ax = plt.subplots(1, 1, tight_layout=True)
ax.imshow(attn_image.as_array()[attn_image.shape[0] // 2, :, :], cmap="Greys")
fig.show()

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

# %%
# read the normalization sinogram
# -------------------------------

# create acquisition sensitivity model from normalisation data
asm_norm = sirf.STIR.AcquisitionSensitivityModel(norm_file)

asm = sirf.STIR.AcquisitionSensitivityModel(asm_norm, asm_attn)
asm.set_up(acq_data)
acq_model.set_acquisition_sensitivity(asm)

# %%
# scatter estimation
# ------------------

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

# %%
# setup the objective function to be maximised
# ---------------------------------------------

# define objective function to be maximized as
# Poisson logarithmic likelihood (with linear model for mean)
obj_fun = (
    sirf.STIR.PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin()
)
obj_fun.set_acquisition_model(acq_model)
obj_fun.set_acquisition_data(listmode_data)
obj_fun.set_num_subsets(num_subsets)

# %%
# calculate the gradient of the (subset) objective function w.r.t. to an image
# ----------------------------------------------------------------------------

print("calculating the gradient of the objective function")
initial_image = acq_data.create_uniform_image(value=1, xy=nxny)
obj_fun.set_up(initial_image)

# the subset 0 objective function w.r.t. to the initial image
grad0 = obj_fun.gradient(initial_image, 0)
#
sens_img_0 = obj_fun.get_subset_sensitivity(0)

## %%
#
## select Ordered Subsets Maximum A-Posteriori One Step Late as the
## reconstruction algorithm (since we are not using a penalty, or prior, in
## this example, we actually run OSEM);
## this algorithm does not converge to the maximum of the objective function
## but is used in practice to speed-up calculations
## See the reconstruction demos for more complicated examples
#reconstructor = sirf.STIR.OSMAPOSLReconstructor()
#reconstructor.set_objective_function(obj_fun)
#reconstructor.set_num_subsets(num_subsets)
#reconstructor.set_num_subiterations(num_iter)
#
## set up the reconstructor based on a sample image
## (checks the validity of parameters, sets up objective function
## and other objects involved in the reconstruction, which involves
## computing/reading sensitivity image etc etc.)
#print('setting up, please wait...')
#reconstructor.set_up(initial_image)
#
## set the initial image estimate
#reconstructor.set_current_estimate(initial_image)
#
## reconstruct
#print('reconstructing, please wait...')
#reconstructor.process()
#recon = reconstructor.get_output()
#recon.write(output_file)

