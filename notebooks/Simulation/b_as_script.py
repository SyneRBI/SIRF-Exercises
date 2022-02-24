# %% [markdown]
# ## Notbook B: How to simulate MR data with motion artefacts
# 
# #### Prerequisites:
# - basic knowledge of Python 
# - formatted input data
# 
# #### Goals:
# - simulating a 2D MR rawdata set with cardio-respiratory motion
# 
# #### Content overview: 
# - using simulation to store ground truth T1, T2 and spin density maps

# %% [markdown]
# ### Simulation and Dynamics
# The simulation has a 

# %%
from pathlib import Path
import os 

# this is where we store the properly formatted data
root_path = Path(os.getenv("SIRF_INSTALL_PATH"))
root_path = root_path / "share/SIRF-3.1/Simulation/"
fpath_input = root_path / "Input"


# %%

import sirf.DynamicSimulation as pDS
import sirf.Reg as pReg

# set up simulation as we know it from before
fname_xml = fpath_input / "XCAT_TissueParameters_XML.xml"
fname_segmentation = fpath_input / "segmentation.nii"

segmentation = pReg.NiftiImageData3D(str(fname_segmentation))
simulation = pDS.MRDynamicSimulation(segmentation, str(fname_xml))


# %%
# without external signal you can set an SNR
SNR = 10
SNR_label = 13

simulation.set_snr(SNR)
simulation.set_snr_label(SNR_label)

# %%
import sirf.Gadgetron as pMR

fname_contrast_template = fpath_input / "contrast_template.h5"
contrast_template = pMR.AcquisitionData(str(fname_contrast_template))
contrast_template = pMR.preprocess_acquisition_data(contrast_template)

fname_acquisition_template = fpath_input / "acquisition_template.h5"
acquisition_template = pMR.AcquisitionData(str(fname_acquisition_template))

simulation.set_contrast_template_data(contrast_template)



# %%

import numpy as np 

# to activate a golden-angle 2D encoding model we only need to set the trajectory
acquisition_template = pMR.set_goldenangle2D_trajectory(acquisition_template)
simulation.set_acquisition_template_data(acquisition_template)

# compute the CSM from the raw data itself
csm = pMR.CoilSensitivityData()
csm.smoothness = 50
csm.calculate(acquisition_template)

# you can potentially keep 
# however, then you need to set the trajectory above that the rawdata used for sampling.
csm_datatype = csm.as_array().dtype
csm_shape = csm.as_array().shape
unity_csm = np.ones(shape=csm_shape, dtype=csm_datatype)
csm.fill(unity_csm)

simulation.set_csm(csm)

# %%

# we set up our transformation to get a 4-chamber view
offset_x_mm = 0
offset_y_mm = 0
offset_z_mm = -127.5
rotation_angles_deg = [0,0,0]
translation = np.array([offset_x_mm, offset_y_mm, offset_z_mm])
euler_angles_deg = np.array(rotation_angles_deg)

offset_trafo = pReg.AffineTransformation(translation, euler_angles_deg)
simulation.set_offset_trafo(offset_trafo)

# %%
simulation.simulate_data()

# now we simulate and 
fname_output = root_path / "Output/output_b_simulate_motion_static.h5"
if not fname_output.parent.is_dir():
    fname_output.parent.mkdir(parents=True, exist_ok=True)


simulation.write_simulation_results(str(fname_output))

# %%
import auxiliary_functions as aux

simulated_file = pMR.AcquisitionData(str(fname_output))

recon_stat = aux.reconstruct_data(simulated_file)
recon_stat = pReg.NiftiImageData3D(recon_stat)
recon_stat.write('/media/sf_CCPPETMR/tmp_stat.nii')

# %%
import matplotlib.pyplot as plt
f = plt.figure()
plt.imshow(np.transpose(np.squeeze(np.abs(recon_stat.as_array()))))
plt.axis('off')
plt.show()

# %%

import auxiliary_functions as aux

Nt = 10000
t0_s = 0
tmax_s = 60* 5 
f_Hz_resp = 10.2

t_resp, sig_resp = aux.get_normed_surrogate_signal(t0_s, tmax_s, Nt, f_Hz_resp)

plt.figure()
plt.plot(t_resp, sig_resp)
plt.xlabel("time(sec)")
plt.ylabel("sig (a.u.)")

# %% [markdown]
# ### Motion Dynamics
# The motion dynamic consists of two things:
# - a surrogate signal
# - a set of displacement vector fields
# 
# Furthermore you need to define 
# - how many motion states you want to simulate.
# - where the ground truth should be stored to.

# %%
# the motion is awill 


# configure the motion
num_motion_states = 3
# RESP
num_sim_resp_states = num_motion_states
resp_motion = pDS.MRMotionDynamic( num_sim_resp_states )
resp_motion.set_dynamic_signal(t_resp, sig_resp)
resp_motion.set_cyclicality(False)
resp_motion.set_groundtruth_folder_prefix(str(root_path / "Output/gt_resp/"))

aux.set_motionfields_from_path(resp_motion, str(fpath_input / 'mvfs_resp/'))
simulation.add_motion_dynamic(resp_motion)

# %%
simulation.simulate_data()

# now we simulate and 
fname_output = root_path / "Output/output_b_simulate_motion_breathing.h5"
if not fname_output.parent.is_dir():
    fname_output.parent.mkdir(parents=True, exist_ok=True)


simulation.write_simulation_results(str(fname_output))

# %%

simulated_file = pMR.AcquisitionData(str(fname_output))

recon_resp = aux.reconstruct_data(simulated_file)
recon_resp = pReg.NiftiImageData3D(recon_resp)
recon_resp.write('/media/sf_CCPPETMR/tmp_resp.nii')

# %%
textcolor = 'black'
f, axs = plt.subplots(1,3)
axs[0].imshow(np.transpose(np.squeeze(recon_stat.as_array())))
axs[0].axis("off")
axs[0].set_title("Static")
axs[0].title.set_color(textcolor)

axs[1].imshow(np.transpose(np.squeeze(recon_resp.as_array())))
axs[1].axis("off")
axs[1].set_title("Breathing")
axs[1].title.set_color(textcolor)

axs[2].imshow(np.transpose(np.squeeze(recon_resp.as_array() - recon_stat.as_array())))
axs[2].axis("off")
axs[2].set_title("difference")
axs[2].title.set_color(textcolor)

# %% [markdown]
# ### Recap
# In this notebook we 
# - used the simulation to generate a rawdata file with magnetisation based on the template file. 
# - added a motion-dynamic to the simulation to perform respiration during the data acquisition.
# 
# _Next step: use the simulation to simulate an MRF acquisition._

# %% [markdown]
# 


