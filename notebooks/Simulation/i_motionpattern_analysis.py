#!/usr/bin/env python
# coding: utf-8

# ## Notebook I: Analysing effect of different motion patterns
# Goals:
# - we want to run the simulation with different motion patterns and analyse the effect on resulting T1 maps

# In[ ]:


import numpy as np
import auxiliary_functions as aux

import matplotlib.pyplot as plt
from pathlib import Path
import os, time

import sirf.DynamicSimulation as pDS
import sirf.Reg as pReg
import sirf.Gadgetron as pMR

# this is where we store the properly formatted data
root_path = aux.root_path
fpath_input = root_path / "Input"
fpath_output = root_path / "Output"


# In[ ]:


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


# In[ ]:


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
    fpath_fingerprinting = fpath_input / "Fingerprinting"
    mrf_dynamic = get_preconfigured_mrfsignal(fpath_fingerprinting, subset_idx)
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


# In[ ]:


# we always need this setup for the motion, the surrogate will dictate what happens
def get_preconfigured_motiondynamic(fpath_mvfs, surrogate_time, surrogate_signal):
    
    num_binned_states = 10
    motion = pDS.MRMotionDynamic( num_binned_states )
    motion.set_dynamic_signal(surrogate_time, surrogate_signal)
    motion.set_cyclicality(False)
    
    aux.set_motionfields_from_path(motion, fpath_mvfs)

    return motion


# In[ ]:


num_simul_acquisitions = None 
fname_static_sim =  fpath_output / "output_i_simulate_static.h5"

simulation, acquisition_template = get_preconfigured_MRF_simulation_object(fpath_input, num_acquisitions=num_simul_acquisitions)

tstart = time.time()

prefix_ground_truth = fpath_output / "output_i_gt_stat"
fnames_stat_gt = simulation.save_parametermap_ground_truth(str(prefix_ground_truth))

if not fname_static_sim.exists():
    simulation.simulate_data()

    print("--- Required {} minutes for the simulation.".format( (time.time()-tstart)/60))
    fname_static_sim.parent.mkdir(parents=True, exist_ok=True)
    simulation.write_simulation_results(str(fname_static_sim))


# In[ ]:


# General time axis, let the guy move for 10 minutes
Nt = 10000
t0_s = 0
tmax_s = 15

fpath_resp_mvfs = str(fpath_input / 'mvfs_resp/')
fpath_card_mvfs = str(fpath_input / 'mvfs_card/')


# In[ ]:


## and the same drill for the respiration
fname_resp_sim = fpath_output / "output_i_simulate_resp.h5"

f_Hz_resp = 0.25
t_resp, sig_resp = aux.get_normed_sinus_signal(t0_s, tmax_s, Nt, f_Hz_resp)

resp_motion = get_preconfigured_motiondynamic(fpath_resp_mvfs, t_resp, sig_resp)
simulation, acquisition_template = get_preconfigured_MRF_simulation_object(fpath_input, num_acquisitions=num_simul_acquisitions)

simulation.add_motion_dynamic(resp_motion)

prefix_ground_truth = fpath_output / "output_i_gt_resp"
fnames_resp_gt = simulation.save_parametermap_ground_truth(str(prefix_ground_truth))

tstart = time.time()

if not fname_resp_sim.exists():
    simulation.simulate_data()

    print("--- Required {} minutes for the simulation.".format( (time.time()-tstart)/60))
    fname_resp_sim.parent.mkdir(parents=True, exist_ok=True)

    simulation.write_simulation_results(str(fname_resp_sim))


# In[ ]:


## and the same drill for the respiration
fname_half_resp_sim = fpath_output / "output_i_simulate_half_resp_amplitude.h5"

f_Hz_resp = 0.25
t_resp, sig_resp = aux.get_normed_sinus_signal(t0_s, tmax_s, Nt, f_Hz_resp)

half_resp_motion = get_preconfigured_motiondynamic(fpath_resp_mvfs, t_resp,  0.5*sig_resp)
simulation, acquisition_template = get_preconfigured_MRF_simulation_object(fpath_input, num_acquisitions=num_simul_acquisitions)

simulation.add_motion_dynamic(half_resp_motion)

prefix_ground_truth = fpath_output / "output_i_gt_halfresp"
fnames_halfresp_gt = simulation.save_parametermap_ground_truth(str(prefix_ground_truth))

tstart = time.time()

if not fname_half_resp_sim.exists():
    simulation.simulate_data()

    print("--- Required {} minutes for the simulation.".format( (time.time()-tstart)/60))
    fname_half_resp_sim.parent.mkdir(parents=True, exist_ok=True)

    simulation.write_simulation_results(str(fname_half_resp_sim))


# In[ ]:


fname_card_sim = fpath_output / "output_i_simulate_card.h5"

f_Hz_card = 1.25
t_card, sig_card = aux.get_normed_sawtooth_signal(t0_s, tmax_s, Nt, f_Hz_card)

card_motion = get_preconfigured_motiondynamic(fpath_card_mvfs, t_card,  sig_card)
simulation, acquisition_template = get_preconfigured_MRF_simulation_object(fpath_input, num_acquisitions=num_simul_acquisitions)

simulation.add_motion_dynamic(card_motion)

prefix_ground_truth = fpath_output / "output_i_gt_card"
fnames_card_gt = simulation.save_parametermap_ground_truth(str(prefix_ground_truth))

tstart = time.time()
if not fname_card_sim.exists():
    simulation.simulate_data()
    print("--- Required {} minutes for the simulation.".format( (time.time()-tstart)/60))
    fname_card_sim.parent.mkdir(parents=True, exist_ok=True)

    simulation.write_simulation_results(str(fname_card_sim))


# In[ ]:


fname_cardioresp_sim = fpath_output / "output_i_simulate_cardioresp.h5"


simulation, acquisition_template = get_preconfigured_MRF_simulation_object(fpath_input, num_acquisitions=num_simul_acquisitions)

simulation.add_motion_dynamic(resp_motion)
simulation.add_motion_dynamic(card_motion)

prefix_ground_truth = fpath_output / "output_i_gt_cardioresp"
fnames_cardioresp_gt = simulation.save_parametermap_ground_truth(str(prefix_ground_truth))

tstart = time.time()
if not fname_cardioresp_sim.exists():
    simulation.simulate_data()
    print("--- Required {} minutes for the simulation.".format( (time.time()-tstart)/60))
    fname_cardioresp_sim.parent.mkdir(parents=True, exist_ok=True)

    simulation.write_simulation_results(str(fname_cardioresp_sim))


# In[ ]:


# load dictionary
fname_dict = fpath_fingerprinting / "dict_70_1500.npz"
mrfdict = np.load(fname_dict)

dict_theta = mrfdict['dict_theta']
dict_mrf = mrfdict['dict_norm']

# otherwise it's too annoying to wait for this
subsample_dict_factor = 1
dict_theta = dict_theta[0:-1:subsample_dict_factor,:]
dict_mrf= dict_mrf[0:-1:subsample_dict_factor,:]


# In[ ]:



def perform_matching(fname_rawdata, num_recon_recon_states, dictionary, dictionary_parameters):
    
    ad = pMR.AcquisitionData(str(fname_rawdata))
    recon, ad = aux.reconstruct_timeresolved(ad, num_recon_recon_states)
    dict_sliding = np.transpose(aux.apply_databased_sliding_window(ad, np.transpose(dictionary)))
    match = aux.dictionary_matching(recon, dict_sliding, dictionary_parameters)
    
    return match

def perform_gated_matching(fname_rawdata, num_recon_recon_states, dictionary, dictionary_parameters, motiondyn, keep_bins):
    
    ad = pMR.AcquisitionData(str(fname_rawdata))
    idx_corr = motiondyn.get_idx_corr(ad)
    
    recon_gated, ad_gated = aux.reconstruct_timeresolved_gated(ad, num_recon_recon_states, idx_corr, keep_bins)
    
    dict_gated = np.transpose(aux.gate_data(np.transpose(dictionary), idx_corr, keep_bins))
    dict_gated = np.transpose(aux.apply_databased_sliding_window(ad_gated, np.transpose(dict_gated)))

    match = aux.dictionary_matching(recon_gated, dict_gated, dictionary_parameters)
    return match


# In[ ]:


import nibabel as nib

def load_gt(fnames_gt):
    print(fnames_gt)
    t1 = np.transpose(np.squeeze(nib.load(fnames_gt[0]).get_fdata()))
    t2 = np.transpose(np.squeeze(nib.load(fnames_gt[1]).get_fdata()))
    labels =  np.transpose(np.squeeze(nib.load(fnames_gt[3]).get_fdata()))
    
    return labels, t1, t2

def compute_and_store_maps(fname_output, fnames_gt, fname_rawdata, dictionary, dictionary_parameters, gated=False, motiondyn=None, keep_bins=None):
    if gated:
        assert motiondyn is not None, "Please pass a motiondynamic"
        assert keep_bins is not None, "Please pass bins to keep"
        
    num_recon_states = 75
    if gated:
        match = perform_gated_matching(fname_rawdata, num_recon_states, dictionary, dictionary_parameters, motiondyn, keep_bins)
    else:
        match = perform_matching(fname_rawdata, num_recon_states, dictionary, dictionary_parameters)
    
    labels, gt_t1, gt_t2 = load_gt(fnames_gt)
    
    
    match_t1 = np.abs(np.squeeze(match[...,1]))
    match_t2 = np.abs(np.squeeze(match[...,2]))
    
    np.savez(str(fname_output), labels=labels, gt_t1=gt_t1, gt_t2=gt_t2, match_t1=match_t1, match_t2=match_t2)
    


# In[ ]:



fname_output = fpath_output / "output_i_fitresults_stat.npz"
if not fname_output.is_file():
    compute_and_store_maps(fname_output, fnames_stat_gt, fname_static_sim, dict_mrf, dict_theta)

fname_output = fpath_output / "output_i_fitresults_resp.npz"
if not fname_output.is_file():
    compute_and_store_maps(fname_output, fnames_resp_gt, fname_resp_sim, dict_mrf, dict_theta)

fname_output = fpath_output / "output_i_fitresults_card.npz"
if not fname_output.is_file():
    compute_and_store_maps(fname_output, fnames_card_gt, fname_card_sim, dict_mrf, dict_theta)
    
fname_output = fpath_output / "output_i_fitresults_cardioresp.npz"
if not fname_output.is_file():
    compute_and_store_maps(fname_output, fnames_cardioresp_gt, fname_cardioresp_sim, dict_mrf, dict_theta)
    
fname_output = fpath_output / "output_i_fitresults_half_resp.npz"
if not fname_output.is_file():
    compute_and_store_maps(fname_output, fnames_halfresp_gt, fname_half_resp_sim, dict_mrf, dict_theta)
    
fname_output = fpath_output / "output_i_fitresults_gated_card.npz"
if not fname_output.is_file():
    keep_bins =np.arange(5,10)
    compute_and_store_maps(fname_output, fnames_card_gt, fname_card_sim, dict_mrf, dict_theta, gated=True, motiondyn=card_motion, keep_bins=keep_bins)    

