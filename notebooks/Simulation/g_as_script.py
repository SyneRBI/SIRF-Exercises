# %% [markdown]
# ## Notbook G: Simulating motion in an MRF acquisition
# 
# #### Prerequisites:
# - a tissue segmentation and XML file.
# - template rawdata.
# - displacement vector field modeling cardiac and respiratory motion.
# - pre-computed time-resolved magnetisation for tissue.
# 
# #### Goals:
# - performing MRF simulation including respiratory and/or cardiac motion

# %%
from pathlib import Path
import os 
import numpy as np
import auxiliary_functions as aux

import sirf.DynamicSimulation as pDS
import sirf.Reg as pReg
import sirf.Gadgetron as pMR



# this is where we store the properly formatted data
root_path = Path(os.getenv("SIRF_INSTALL_PATH"))
root_path = root_path / "share/SIRF-3.1/Simulation/"
fpath_input = root_path / "Input"


# %%
# import matplotlib.pyplot as plt
from pathlib import Path

fpath_epg_result = Path("/media/sf_CCPPETMR/TestData/Input/xDynamicSimulation/pDynamicSimulation/Fingerprints/")

fname_epg_input = fpath_epg_result / "XCAT_tissue_parameter_list.npz"
fname_epg_simulation = fpath_epg_result / "XCAT_tissue_parameter_fingerprints.npy"

# re-assign non-unique tissue combinations
epg_input = np.load(fname_epg_input)
inverse_idx = epg_input["unique_idx_inverse"]
inverse_idx.shape

#
epg_output = np.load(fname_epg_simulation)
magnetisation = epg_output[:, inverse_idx]


num_sim_acq = magnetisation.shape[0]
magnet_subset = np.arange(num_sim_acq)
magnetisation = magnetisation[magnet_subset, :]
print(magnetisation.shape)

# %%

fname_xml = fpath_input / "XCAT_TissueParameters_XML.xml"
fname_segmentation = fpath_input / "segmentation.nii"
segmentation = pReg.NiftiImageData3D(str(fname_segmentation))

simulation = pDS.MRDynamicSimulation(segmentation, str(fname_xml))

# %%
fname_acquisition_template = fpath_input / "acquisition_template.h5"
acquisition_template = pMR.AcquisitionData(str(fname_acquisition_template))

num_acquisitions = magnetisation.shape[0]
subset_idx = np.arange(num_acquisitions)
acquisition_template = acquisition_template.get_subset(subset_idx)

acquisition_template = pMR.set_goldenangle2D_trajectory(acquisition_template)

simulation.set_template_data(acquisition_template)

csm = aux.gaussian_2D_coilmaps(acquisition_template)
simulation.set_csm(csm)

# %%
# we add the usual offset transformation
offset_x_mm = 0
offset_y_mm = 0
offset_z_mm = -14
rotation_angles_deg = [0,0,0]
translation = np.array([offset_x_mm, offset_y_mm, offset_z_mm])
euler_angles_deg = np.array(rotation_angles_deg)

offset_trafo = pReg.AffineTransformation(translation, euler_angles_deg)
simulation.set_offset_trafo(offset_trafo)

prefix_output = root_path / "Output/output_g_ground_truth"
simulation.save_parametermap_ground_truth(str(prefix_output))

# %% [markdown]
# To set up a time-dependant magnetisation we use the same principle as with the motion dynamic:
# - first we set up a dynamic object
# - we set the required parameters
# - we add the dynamic to the simulation

# %%
# now we need to create an external conrast dynamic

# say which labels are in the dynamic
signal_labels = np.arange(magnetisation.shape[1])
magnetisation = np.transpose(magnetisation)

mrf_signal = pDS.ExternalMRSignal(signal_labels, magnetisation)
mrf_dynamic = pDS.ExternalMRContrastDynamic()
mrf_dynamic.add_external_signal(mrf_signal)
simulation.add_external_contrast_dynamic(mrf_dynamic)

# %%
# General time axis, let the guy move for 10 minutes
Nt = 10000
t0_s = 0
tmax_s = 60*10

# %%

## and the same drill for the respiration

f_Hz_resp = 0.25
t_resp, sig_resp = aux.get_normed_sinus_signal(t0_s, tmax_s, Nt, f_Hz_resp)

num_sim_resp_states = 1
resp_motion = pDS.MRMotionDynamic( num_sim_resp_states )
resp_motion.set_dynamic_signal(t_resp, sig_resp)
resp_motion.set_cyclicality(False)
resp_motion.set_groundtruth_folder_prefix(str(root_path / "Output/gt_resp_mrf/"))

aux.set_motionfields_from_path(resp_motion, str(fpath_input / 'mvfs_resp/'))
simulation.add_motion_dynamic(resp_motion)

# now we simulate and store it
import time
tstart = time.time()
simulation.simulate_data()
print("--- Required {} minutes for the simulation.".format( (time.time()-tstart)/60))

fname_output = root_path / "Output/output_g_simulate_mrf_resp.h5"
if not fname_output.parent.is_dir():
    fname_output.parent.mkdir(parents=True, exist_ok=True)

simulation.write_simulation_results(str(fname_output))

# %%

## 
f_Hz_card = 1.25
t_card, sig_card = aux.get_normed_sawtooth_signal(t0_s, tmax_s, Nt, f_Hz_card)

# this number does not matter since every readout get's its own motionstate
num_sim_card_states = 1
card_motion = pDS.MRMotionDynamic( num_sim_card_states )
card_motion.set_dynamic_signal(t_card, sig_card)
card_motion.set_cyclicality(False)
card_motion.set_groundtruth_folder_prefix(str(root_path / "Output/gt_card_mrf/"))

aux.set_motionfields_from_path(card_motion, str(fpath_input / 'mvfs_card/'))
simulation.add_motion_dynamic(card_motion)

# now we simulate and store it
import time
tstart = time.time()
simulation.simulate_data()
print("--- Required {} minutes for the simulation.".format( (time.time()-tstart)/60))

fname_output = root_path / "Output/output_g_simulate_mrf_cardio_resp.h5"
if not fname_output.parent.is_dir():
    fname_output.parent.mkdir(parents=True, exist_ok=True)

simulation.write_simulation_results(str(fname_output))

# %% [markdown]
# ### Recap 
# In this notebook we combined motion dynamics with external contrast dynamics to simulate motion during MRF.
# 
# _Up next: dictionary matching for the simulated data._

# %% [markdown]
# 


