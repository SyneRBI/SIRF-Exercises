# %% [markdown]
# ### Notebook C: 

# %% [markdown]
# ## Notbook C: How to preprocess input data for simulation
# 
# #### Prerequisites:
# - basic knowledge of Python 
# - segmentation and motion fields
# - contrast and acquisition template rawdata
# - simulated magnetisation for all tissue types in the segmentation
# 
# #### Goals:
# - Passing a pre-computed magnetisation to the simulation.
# 
# #### Content overview: 
# - extracting the tissue parameters from an XML file.
# - adding a time-dependent magnetisation to the simulation.
# - using a pre-computed magnetisation to perform an MRF simulation.

# %%
from pathlib import Path
import os 

# this is where we store the properly formatted data
root_path = Path(os.getenv("SIRF_INSTALL_PATH"))
root_path = root_path / "share/SIRF-3.1/Simulation/"
fpath_input = root_path / "Input"


# %% [markdown]
# #### Generating Tissue Dictionary
# 
# In the XML file we have defined the MR parameters T1, T2 and spin density for every tissue appearing in our segmentation. If we want to perform an EPG simulation for each these combinations we have to read .
# 
# This can be done using XML parsing and is encapsulated in the TissueParameterList.
# Each row contains the MR tissue parameters in the order: 
# 
# _(label, spin density (% of water), T1(ms), T2(ms), chemical shift (ppm))_
# 
# Once this information is available it can be passed to any external tool.

# %%
import numpy as np
import auxiliary_functions as aux
import TissueParameterList as TPL

fname_xml = fpath_input / "XCAT_TissueParameters_XML.xml"
tpl = TPL.TissueParameterList()
tpl.parse_xml(str(fname_xml))
print("The tissue parameters for the different labels are: \n {}".format(tpl.mr_as_array().astype(np.int)))


# %% [markdown]
# The EPG simulation part is ommitted at this point and a pre-computed magnetisation is loaded from a file.

# %%
import matplotlib.pyplot as plt
from pathlib import Path
import shutil 

fpath_epg_result = Path("/media/sf_CCPPETMR/TestData/Input/xDynamicSimulation/pDynamicSimulation/Fingerprints/")

fname_epg_input = fpath_epg_result / "XCAT_tissue_parameter_list.npz"
fname_epg_simulation = fpath_epg_result / "XCAT_tissue_parameter_fingerprints.npy"

# copy for reference later
shutil.copy( str(fname_epg_input), str(fpath_input))
shutil.copy( str(fname_epg_simulation), str(fpath_input))

# re-assign non-unique tissue combinations
epg_input = np.load(fname_epg_input)
inverse_idx = epg_input["unique_idx_inverse"]
inverse_idx.shape

#
epg_output = np.load(fname_epg_simulation)
magnetisation = epg_output[:, inverse_idx]

plt.figure()
plt.plot(np.abs(epg_output))
plt.xlabel("readout")
plt.ylabel("signal (a.u.)")
plt.show()


# %%
# We go through our usual drill of setting up our simulation

import sirf.DynamicSimulation as pDS
import sirf.Reg as pReg
import sirf.Gadgetron as pMR


fname_segmentation = fpath_input / "segmentation.nii"
segmentation = pReg.NiftiImageData3D(str(fname_segmentation))

simulation = pDS.MRDynamicSimulation(segmentation, str(fname_xml))



# %%

fname_contrast_template = fpath_input / "contrast_template.h5"
contrast_template = pMR.AcquisitionData(str(fname_contrast_template))
contrast_template = pMR.preprocess_acquisition_data(contrast_template)

simulation.set_contrast_template_data(contrast_template)

# %%

fname_acquisition_template = fpath_input / "acquisition_template.h5"
acquisition_template = pMR.AcquisitionData(str(fname_acquisition_template))

num_acquisitions = 128
subset_idx = np.arange(num_acquisitions)
acquisition_template = acquisition_template.get_subset(subset_idx)
acquisition_template = pMR.set_goldenangle2D_trajectory(acquisition_template)

simulation.set_acquisition_template_data(acquisition_template)

# %%
csm = aux.unity_coilmaps_from_rawdata(acquisition_template)
simulation.set_csm(csm)

# %%
# we add the usual simulation

offset_x_mm = 0
offset_y_mm = 0
offset_z_mm = -127.5
rotation_angles_deg = [0,0,0]
translation = np.array([offset_x_mm, offset_y_mm, offset_z_mm])
euler_angles_deg = np.array(rotation_angles_deg)

offset_trafo = pReg.AffineTransformation(translation, euler_angles_deg)
simulation.set_offset_trafo(offset_trafo)

# %% [markdown]
# To set up a time-dependant magnetisation we use the same principle as with the motion dynamic:
# - first we set up a dynamic object
# - we set the required parameters
# - we add the dynamic to the simulation

# %%
# now we need to create an external conrast dynamic

# say which labels are in the dynamic
signal_labels = np.arange(magnetisation.shape[1])
magnetisation = np.transpose(magnetisation[:num_acquisitions,:])

mrf_signal = pDS.ExternalMRSignal(signal_labels, magnetisation)
mrf_dynamic = pDS.ExternalMRContrastDynamic()
mrf_dynamic.add_external_signal(mrf_signal)
simulation.add_external_contrast_dynamic(mrf_dynamic)

# %%
# now we simulate and store it
import time
tstart = time.time()
simulation.simulate_data()
print("--- Required {} minutes for the simulation.".format( (time.time()-tstart)/60))

fname_output = root_path / "Output/output_c_simulate_mrf_static.h5"
if not fname_output.parent.is_dir():
    fname_output.parent.mkdir(parents=True, exist_ok=True)

simulation.write_simulation_results(str(fname_output))

# %%
simulated_data = pMR.AcquisitionData(str(fname_output))
recon = pReg.NiftiImageData3D(aux.reconstruct_data(simulated_data))
recon.write("/media/sf_CCPPETMR/tmp_mrf.nii")


