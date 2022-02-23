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

fname_acquisition_template = fpath_input / "acquisition_template.h5"
acquisition_template = pMR.AcquisitionData(str(fname_acquisition_template))

simulation.set_acquisition_template_data(contrast_template)
simulation.set_acquisition_template_data(acquisition_template)


# %%

import numpy as np 

# to activate a golden-angle 2D encoding model we only need to set the trajectory
acquisition_template = pMR.set_goldenangle2D_trajectory(acquisition_template)

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
rotation_angles_deg = [-15,-15,0]
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
recon = aux.reconstruct_data(simulated_file)

# %%
import matplotlib.pyplot as plt
plt.imshow(np.squeeze(np.abs(recon.as_array())))

# %% [markdown]
# 


