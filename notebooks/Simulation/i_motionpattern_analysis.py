#!/usr/bin/env python
# coding: utf-8

# ## Notebook I: Analysing effect of different motion patterns
# Goals:
# - we want to run the simulation with different motion patterns and analyse the effect on resulting T1 maps

# In[1]:


from pathlib import Path
import os, time
import numpy as np
import auxiliary_functions as aux

import sirf.DynamicSimulation as pDS
import sirf.Reg as pReg
import sirf.Gadgetron as pMR

# this is where we store the properly formatted data
root_path = Path(os.getenv("SIRF_INSTALL_PATH"))
root_path = root_path / "share/SIRF-3.1/Simulation/"
fpath_input = root_path / "Input"

fname_cardiac_sim = root_path / "Output/output_i_simulate_card.h5"
fname_half_resp_sim = root_path / "Output/output_i_simulate_half_resp_amplitude.h5"


# In[2]:


# load dictionary
fname_dict = Path("/media/sf_CCPPETMR/TestData/Input/xDynamicSimulation/pDynamicSimulation/Fingerprints/dict_70_1500.npz")
mrfdict = np.load(fname_dict)

dict_theta = mrfdict['dict_theta']
dict_mrf = mrfdict['dict_norm']

# otherwise it's too annoying to wait for this
subsample_dict_factor = 10
dict_theta = dict_theta[0:-1:subsample_dict_factor,:]
dict_mrf= dict_mrf[0:-1:subsample_dict_factor,:]


# In[3]:


# we always need the same MRF signal
def get_preconfigured_mrfsignal(fpath_epg_result, subset_idx=None):
    
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

    # say which labels are in the dynamic
    signal_labels = np.arange(magnetisation.shape[1])
    magnetisation = np.transpose(magnetisation)

    if subset_idx is not None:
        magnetisation = magnetisation[:,subset_idx]

    mrf_signal = pDS.ExternalMRSignal(signal_labels, magnetisation)
    mrf_dynamic = pDS.ExternalMRContrastDynamic()
    mrf_dynamic.add_external_signal(mrf_signal)

    return mrf_dynamic


# In[4]:


# we always need the same     
def get_preconfigured_MRF_simulation_object(fpath_input, num_acquisitions=None):

    fname_xml = fpath_input / "XCAT_TissueParameters_XML.xml"
    fname_segmentation = fpath_input / "segmentation.nii"
    segmentation = pReg.NiftiImageData3D(str(fname_segmentation))

    simulation = pDS.MRDynamicSimulation(segmentation, str(fname_xml))

    fname_acquisition_template = fpath_input / "acquisition_template.h5"
    acquisition_template = pMR.AcquisitionData(str(fname_acquisition_template))

    if num_acquisitions is None:
        num_acquisitions = 1500#acquisition_template.number()
    
    subset_idx = np.arange(num_acquisitions)
    acquisition_template = acquisition_template.get_subset(subset_idx)
    acquisition_template = pMR.set_goldenangle2D_trajectory(acquisition_template)

    simulation.set_template_data(acquisition_template)
    
    #
    csm = aux.gaussian_2D_coilmaps(acquisition_template)
    simulation.set_csm(csm)

    # add MRF
    fpath_epg_result = Path("/media/sf_CCPPETMR/TestData/Input/xDynamicSimulation/pDynamicSimulation/Fingerprints/")
    mrf_dynamic = get_preconfigured_mrfsignal(fpath_epg_result, subset_idx)
    simulation.add_external_contrast_dynamic(mrf_dynamic)

    # we add the usual offset transformation
    offset_x_mm = 0
    offset_y_mm = 0
    offset_z_mm = -9
    rotation_angles_deg = [0,0,0]
    translation = np.array([offset_x_mm, offset_y_mm, offset_z_mm])
    euler_angles_deg = np.array(rotation_angles_deg)

    offset_trafo = pReg.AffineTransformation(translation, euler_angles_deg)
    simulation.set_offset_trafo(offset_trafo)

    return simulation, acquisition_template


# In[5]:


# we always need this setup for the motion, the surrogate will dictate what happens
def get_preconfigured_motiondynamic(fpath_mvfs, surrogate_time, surrogate_signal):
    
    num_binned_states = 10
    motion = pDS.MRMotionDynamic( num_binned_states )
    motion.set_dynamic_signal(surrogate_time, surrogate_signal)
    motion.set_cyclicality(False)
    
    aux.set_motionfields_from_path(motion, fpath_mvfs)

    return motion


# In[6]:


#
num_simul_acquisitions = None

# General time axis, let the guy move for 10 minutes
Nt = 10000
t0_s = 0
tmax_s = 15

fpath_resp_mvfs = str(fpath_input / 'mvfs_resp/')
fpath_card_mvfs = str(fpath_input / 'mvfs_card/')


# In[7]:


## and the same drill for the respiration
f_Hz_resp = 0.25
t_resp, sig_resp = aux.get_normed_sinus_signal(t0_s, tmax_s, Nt, f_Hz_resp)

half_resp_motion = get_preconfigured_motiondynamic(fpath_resp_mvfs, t_resp,  0.5*sig_resp)
simulation, acquisition_template = get_preconfigured_MRF_simulation_object(fpath_input, num_acquisitions=num_simul_acquisitions)

simulation.add_motion_dynamic(half_resp_motion)


# In[8]:


tstart = time.time()
simulation.simulate_data()

print("--- Required {} minutes for the simulation.".format( (time.time()-tstart)/60))

if not fname_half_resp_sim.parent.is_dir():
    fname_half_resp_sim.parent.mkdir(parents=True, exist_ok=True)

simulation.write_simulation_results(str(fname_half_resp_sim))


# In[9]:


import scipy.signal as scisig
import matplotlib.pyplot as plt
## and the same drill for the respiration
f_Hz_card = 1.25
t_card, sig_card = aux.get_normed_sawtooth_signal(t0_s, tmax_s, Nt, f_Hz_card)

card_motion = get_preconfigured_motiondynamic(fpath_card_mvfs, t_card,  sig_card)
simulation, acquisition_template = get_preconfigured_MRF_simulation_object(fpath_input, num_acquisitions=num_simul_acquisitions)

simulation.add_motion_dynamic(card_motion)


# In[ ]:


tstart = time.time()
simulation.simulate_data()
print("--- Required {} minutes for the simulation.".format( (time.time()-tstart)/60))

if not fname_cardiac_sim.parent.is_dir():
    fname_cardiac_sim.parent.mkdir(parents=True, exist_ok=True)

simulation.write_simulation_results(str(fname_cardiac_sim))


# In[ ]:


resp_data = pMR.AcquisitionData(str(fname_half_resp_sim))

num_time_points = 75
recon_halfresp, __ = aux.reconstruct_timeresolved(resp_data, num_time_points)
aux.write_nii('/media/sf_CCPPETMR/tmp_halfresp.nii', recon_halfresp)


# In[ ]:


cardiac_data = pMR.AcquisitionData(str(fname_cardiac_sim))
num_time_points = 75


# In[ ]:


recon_card, ad_card = aux.reconstruct_timeresolved(cardiac_data, num_time_points)
aux.write_nii('/media/sf_CCPPETMR/tmp_card.nii', recon_card)


# In[ ]:


dict_card = np.transpose(aux.apply_databased_sliding_window(ad_card, np.transpose(dict_mrf)))
match_card = aux.dictionary_matching(recon_card, dict_card, dict_theta)


# In[ ]:


f,ax = plt.subplots(1,2)

ax[0].imshow(np.abs(match_card[:,:,1]),cmap='jet',vmin=0,vmax=2500)
ax[0].axis('off')
ax[0].set_title("T1 Card")

ax[1].imshow(np.abs(match_card[:,:,2]),cmap='magma',vmin=0,vmax=150)
ax[1].axis('off')
ax[1].set_title("T2 Card")


# In[ ]:


cardiac_data = pMR.AcquisitionData(str(fname_cardiac_sim))
idx_corr = card_motion.get_idx_corr(cardiac_data)
keep_bins =np.arange(5,10)

recon_card_gated, ad_cardgated = aux.reconstruct_timeresolved_gated(cardiac_data, num_time_points, idx_corr, keep_bins)
aux.write_nii('/media/sf_CCPPETMR/tmp_cardgated.nii', recon_card_gated)


# In[ ]:


gated_dict = np.transpose(aux.gate_data(np.transpose(dict_mrf), idx_corr, keep_bins))
dict_card_gated = np.transpose(aux.apply_databased_sliding_window(ad_cardgated, np.transpose(gated_dict)))


# In[ ]:


match_cardiogated = aux.dictionary_matching(recon_card_gated, dict_card_gated, dict_theta)


# In[ ]:


f,ax = plt.subplots(1,2)

ax[0].imshow(np.abs(match_cardiogated[:,:,1]),cmap='jet',vmin=0,vmax=2500)
ax[0].axis('off')

ax[1].imshow(np.abs(match_cardiogated[:,:,2]),cmap='magma',vmin=0,vmax=150)
ax[1].axis('off')


# In[ ]:


f,ax = plt.subplots(2,2)

ax[0,0].imshow(np.abs(match_card[:,:,1]),cmap='jet',vmin=0,vmax=2500)
ax[0,0].axis('off')
ax[0,0].set_title('T1')


ax[1,0].imshow(np.abs(match_card[:,:,2]),cmap='magma',vmin=0,vmax=150)
ax[1,0].axis('off')
ax[1,0].set_title('T2')

ax[0,1].imshow(np.abs(match_cardiogated[:,:,1]),cmap='jet',vmin=0,vmax=2500)
ax[0,1].axis('off')
ax[0,1].set_title('T1 Gated')

ax[1,1].imshow(np.abs(match_cardiogated[:,:,2]),cmap='magma',vmin=0,vmax=150)
ax[1,1].axis('off')
ax[1,1].set_title('T2 Gated')

plt.savefig("/media/sf_CCPPETMR/fig_i_gating.png", dpi=300)

