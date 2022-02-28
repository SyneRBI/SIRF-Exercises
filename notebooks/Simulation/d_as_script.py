# %% [markdown]
# ## Notbook D: Dictionary Matching
# 
# #### Prerequisites:
# - a simulated rawdata file
# - a simulated dictionary for matching.
# 
# #### Goals:
# - reconstruct time-resolved images containing the MRF signal.
# - extract T1, T2 and spin density maps from the 
# 
# #### Content overview: 
# - processing the simualted data to contain time-information
# - reconstructing the data using an inverse FFT
# - perform dictionary matching
# - compare with ground truth parameter maps.

# %%
from pathlib import Path
import os 

# this is where we store the properly formatted data
root_path = Path(os.getenv("SIRF_INSTALL_PATH"))
root_path = root_path / "share/SIRF-3.1/Simulation/"
fpath_input = root_path / "Output"

# %%
import sirf.Gadgetron as pMR
import auxiliary_functions as aux

# fname_simulated_file = fpath_input / "output_c_simulate_mrf_static.h5"
fname_simulated_file = fpath_input / "output_c_simulate_mrf_static_64.h5"
ad = pMR.AcquisitionData(str(fname_simulated_file))

# %%
# a quick parse of the header into a dictionary
# allows us to modify it quickly such that the reconstruction
# can pick up that we want the reconstruction to be time-resolved
import xmltodict

hdr = ad.get_header()
doc = xmltodict.parse(hdr)
doc['ismrmrdHeader']['encoding']['encodingLimits']['repetition']['minimum'] = 0
doc['ismrmrdHeader']['encoding']['encodingLimits']['repetition']['center'] = 0
doc['ismrmrdHeader']['encoding']['encodingLimits']['repetition']['maximum'] = ad.number()

hdr = xmltodict.unparse(doc)

# %%
# now we make a new container to fill in the modified repetitions
ad_resolved = ad.new_acquisition_data()
ad_resolved.set_header(hdr)



# %%
# this way we will reconstruct one image per readout
for ia in range(ad.number()):    
    acq = ad.acquisition(ia)
    acq.set_repetition( ia )
    ad_resolved.append_acquisition(acq)

ad_resolved.sort_by_time()

ad = pMR.set_goldenangle2D_trajectory(ad_resolved)
ad_resolved = pMR.set_goldenangle2D_trajectory(ad_resolved)

csm = aux.unity_coilmaps_from_rawdata(ad_resolved)

# %%
import numpy as np
import auxiliary_functions as aux


import time
tstart = time.time()
recon = aux.reconstruct_data(ad_resolved, csm)
print("--- Required {} seconds for reconstruction.".format( (time.time()-tstart)/60))

import nibabel as nib
img = nib.Nifti1Image(np.abs(recon.as_array()), np.eye(4))
nib.save(img,"/media/sf_CCPPETMR/tmp_mrfresolved.nii")

# %%



