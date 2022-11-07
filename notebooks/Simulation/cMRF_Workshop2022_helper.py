import shutil, os
from pathlib import Path 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import subprocess 

import nibabel as nib
import numpy as np

import sirf.Reg as pReg
import sirf.DynamicSimulation as pDS
import sirf.Gadgetron as pMR
from cil.utilities.jupyter import islicer, link_islicer

import auxiliary_functions as aux
import TissueParameterList as TPL

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Set up folders for simulation (and remove exisiting ones)
def clear_folders(fpath_list):
    for fpath in fpath_list:
        if os.path.isdir(fpath):
            user_in = input(f"Remove {fpath} and all it's content? (y/n):")
            if user_in.lower() == 'y':
                shutil.rmtree(fpath)
                fpath.mkdir(parents=True)
                print(f"{fpath} was removed and recreated")
            else:
                print(f"{fpath} was not removed")
              
            
def vis_mrf_im_data(im_dat):
    f, ax = plt.subplots(1, 3, figsize=(10, 3))
    title_list = ['Image', 'Temporal view', 'Average']
    for ind in range(3):
        if ind == 0: # plot the image with the highest signal
            im_sig = np.sum(im_dat, axis=(1,2))
            cidx = np.argmax(im_sig)
            cim = im_dat[cidx,:,:].transpose()
        elif ind == 1: # plot a space-time view
            cim = im_dat[:,:,im_dat.shape[2]//2].transpose()
        else: # plot the average
            cim = np.average(im_dat, axis=0).transpose()
        ax[ind].imshow(cim, aspect='auto')
        ax[ind].set_xticks([])
        ax[ind].set_yticks([])
        ax[ind].set_title(title_list[ind])
                
def vis_m0_t1_t2_mrf_maps(t1_t2_m0_map_list, method_titles=None):

    if method_titles is not None:
        assert len(t1_t2_m0_map_list) == len(method_titles), 'Method_titles should be list of the same length as t1_t2_m0_map_list'
        
    num_maps = len(t1_t2_m0_map_list)
    f, ax = plt.subplots(num_maps, 3, figsize=(10, 3*num_maps), squeeze=False)
    textcolor = 'black'
    
    cmap_list = ['gray', 'jet', 'magma']
    vmax_list = [None, 2500, 150]
    title_list = ['M0', 'T1 (ms)', 'T2 (ms)']
    
    for ind in range(num_maps):
        for jnd in range(3): # T1, T2, M0
            divider = make_axes_locatable(ax[ind, jnd])
            cax = divider.append_axes('right', size='5%', pad=0.05)

            if vmax_list[jnd] is None:
                im = ax[ind, jnd].imshow(np.transpose(np.abs(t1_t2_m0_map_list[ind][:,:,jnd])), 
                                         cmap=cmap_list[jnd], vmin=0)
            else:
                im = ax[ind, jnd].imshow(np.transpose(np.abs(t1_t2_m0_map_list[ind][:,:,jnd])), 
                                         cmap=cmap_list[jnd], vmin=0, vmax=vmax_list[jnd])
            ax[ind, jnd].set_xticks([])
            ax[ind, jnd].set_yticks([])
            ax[ind, jnd].set_title(title_list[jnd])
            ax[ind, jnd].title.set_color(textcolor)
            
            if method_titles is not None:
                if jnd == 0:
                    ax[ind, jnd].set_ylabel(method_titles[ind])
                
            cbar = f.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(title_list[jnd])
            

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()
    
    
def load_gt_maps(filenames_parametermaps):
    for ind in range(len(filenames_parametermaps)):
        curr_gt = nib.load(filenames_parametermaps[ind]).get_fdata()
        if ind == 0:
            gt_maps = np.zeros(np.squeeze(curr_gt).shape + (len(filenames_parametermaps),))
        gt_maps[:,:,ind] = np.squeeze(curr_gt)
    # Resort from T1, T2, M0 to M0, T1, T2
    gt_maps = gt_maps[:,:,[2,0,1]]
    # Adapt orientation
    gt_maps = np.moveaxis(gt_maps, (0,1,2), (1,0,2))
    return(gt_maps)
    
                
# cMRF_Workshop2022_1
def prep_sim(fpath_segmentation_nii, fpath_resp_mvf, fpath_card_mvf, fpath_in, num_slices=10, slice_start=35):
    # Load segmentation
    print(f'Loading segmentation from {str(fpath_segmentation_nii)}.')
    segmentation_nii = nib.load(str(fpath_segmentation_nii))
    segmentation = segmentation_nii.get_fdata()
    
    # Select small number of slices to speed up simulation
    slice_end = slice_start + num_slices
    slice_subset = range(slice_start,slice_end)
    segmentation = segmentation[:,:,slice_subset]
    
    # Save as nii with correct header information
    resolution_mm_per_pixel = np.array([2,2,-2,1])

    # The first voxel center is lies at -FOV / 2 + dx/2
    offset_mm =(-np.array(segmentation.shape)/2 + 0.5) * resolution_mm_per_pixel[0:3]

    affine = np.diag(resolution_mm_per_pixel)
    affine[:3,3] = offset_mm

    img = nib.Nifti1Image(segmentation, affine)
    img.set_qform(affine) # crucial!
    fname_segmentation = fpath_in / "segmentation.nii"
    nib.save(img, str(fname_segmentation))
    
    # Cardiac motion fields
    card_mvfs = aux.read_nii_motionfields(str(fpath_card_mvf))
    card_mvfs = card_mvfs[:,:,:,slice_subset,:,:]

    fpath_card_mvf_output = fpath_in / 'mvfs_card'
    fpath_card_mvf_output.mkdir(exist_ok=True,parents=True)
    aux.store_nii_mvfs(str(fpath_card_mvf_output), card_mvfs, affine)
    
    # Respiratory motion fields
    resp_mvfs = aux.read_nii_motionfields(str(fpath_resp_mvf))
    resp_mvfs = resp_mvfs[:,:,:,slice_subset,:,:]

    fpath_resp_mvf_output = fpath_in / 'mvfs_resp'
    fpath_resp_mvf_output.mkdir(exist_ok=True,parents=True)
    aux.store_nii_mvfs(str(fpath_resp_mvf_output), resp_mvfs, affine)
    
    return(fname_segmentation, fpath_card_mvf_output, fpath_resp_mvf_output)
    
    
# cMRF_Workshop2022_2
def set_up_mrf_sim(fname_acquisition_template, fname_segmentation, fname_xml, fname_epg_par, fname_epg_sig, prefix_ground_truth, num_acquisitions=1500, acq_step=1):
    # Load the labels into SIRF Nifti Container
    segmentation = pReg.NiftiImageData3D(str(fname_segmentation))
    
    # Load precalculated MRF signals
    epg_input = np.load(fname_epg_par)
    inverse_idx = epg_input["unique_idx_inverse"]
    signal_labels = epg_input["labels"]
    epg_output = np.load(fname_epg_sig)
    magnetisation = epg_output[inverse_idx, :]
    
    magnetisation = np.moveaxis(magnetisation, (0,1), (1,0))
    
    # Set up the simulation with the segmentation and corresponding XML filename.
    if fname_xml.exists():
        mrsim = pDS.MRDynamicSimulation(segmentation, str(fname_xml))
    else:
        raise AssertionError("You didn't provide the XML file {} we were looking for.".format(fname_xml))
        
    # Set up acquisition template
    acquisition_template = pMR.AcquisitionData(str(fname_acquisition_template))

    subset_idx = np.arange(int(num_acquisitions/acq_step))
    acquisition_template = acquisition_template.get_subset(subset_idx)
    acquisition_template = pMR.set_goldenangle2D_trajectory(acquisition_template)
    mrsim.set_template_data(acquisition_template)
    
    # Now we set up an affine transformation to move the acquired slice around
    offset_x_mm = 0
    offset_y_mm = 0
    offset_z_mm = -9
    rotation_angles_deg = [0,0,0]
    translation = np.array([offset_x_mm, offset_y_mm, offset_z_mm])
    euler_angles_deg = np.array(rotation_angles_deg)

    offset_trafo = pReg.AffineTransformation(translation, euler_angles_deg)
    mrsim.set_offset_trafo(offset_trafo)
    
    # Set up Gaussian noise
    SNR = 10
    SNR_label = 13
    mrsim.set_snr(SNR)
    mrsim.set_snr_label(SNR_label)
    
    # Get gaussian coilmaps
    csm = aux.gaussian_2D_coilmaps(acquisition_template)
    csm_arr = csm.as_array()
    mrsim.set_csm(csm)
    
    # MRF signal dynamic
    magnetisation = np.transpose(magnetisation[0:num_acquisitions:acq_step,:])

    mrf_signal = pDS.ExternalMRSignal(signal_labels, magnetisation)
    mrf_dynamic = pDS.ExternalMRContrastDynamic()
    mrf_dynamic.add_external_signal(mrf_signal)
    mrsim.add_external_contrast_dynamic(mrf_dynamic)
    
    return(mrsim, magnetisation)


def rec_mrf_im(fname_sim_data, num_recon_imgs=250):
    simulated_data = pMR.AcquisitionData(str(fname_sim_data))
    print(f"We have {simulated_data.number()} acquisitions")
    # First compute the CSM based on all data
    csm = pMR.CoilSensitivityData()
    csm.smoothness = 50
    csm.calculate(simulated_data)
    csm_arr = csm.as_array()

    # Set up dynamic time-resolved reconstruction
    simulated_data = aux.activate_timeresolved_reconstruction(simulated_data, num_recon_imgs)

    # Set up a new CSM based on the time-resolved acquisition data
    csm = pMR.CoilSensitivityData()
    csm.calculate(simulated_data) 
    
    # We want to use the coilmap that was computed from the entire dataset
    # so we give every repetition the same coilmap
    num_reps = csm.as_array().shape[1]
    csm_arr = np.tile(csm_arr, (1,num_reps,1,1))

    # Unfortunately these two axes have to be swapped.
    csm_arr = np.swapaxes(csm_arr, 0, 1)
    csm = csm.fill(csm_arr.astype(csm.as_array().dtype))
    
    # Carry out reconstruction
    recon = aux.reconstruct_data(simulated_data, csm)
    
    return(recon, simulated_data)


def match_sig(recon, simulated_data, fname_dict, dict_us_factor = 10, num_acquisitions=1500, acq_step=1):
    mrfdict = np.load(fname_dict)

    dict_theta = mrfdict['dict_theta']

    dict_mrf = mrfdict['dict_norm']
    dict_mrf = dict_mrf[:,0:num_acquisitions:acq_step]
    dict_mrf = np.transpose( aux.apply_databased_sliding_window(simulated_data, np.transpose(dict_mrf)))

    dict_mrf = dict_mrf[0:-1:dict_us_factor,:]
    dict_theta = dict_theta[0:-1:dict_us_factor,:]
    
    print("Our dictionary to match is of size {}".format(dict_mrf.shape))
    img_series = recon.as_array()
    img_shape = img_series.shape[1:]
    img_series_1d = np.transpose(np.reshape(img_series,(img_series.shape[0], -1)))
    
    # This checks the largest overlap between time-profile and dictionary entries
    # If the RAM overflows this will catch it and perform the task in multiple sets.
    dict_match = aux.match_dict(dict_mrf, dict_theta, img_series_1d)
    dict_match = np.reshape(dict_match, (*img_shape, -1))
    
    return(dict_match)

    
def get_t1_t2_for_epg(fname_xml, fname_par_epg):
    # Load xml parameter file
    tpl = TPL.TissueParameterList()
    tpl.parse_xml(str(fname_xml))

    tplist = tpl.mr_as_array()
    
    # Get only unique parameter combinations
    tpunique, inverse_idx = np.unique(tplist[:,1:], axis=0, return_inverse=True)
    
    # Save as npz array
    np.savez(fname_par_epg, mr_parameters=tpunique, unique_idx_inverse=inverse_idx, mr_params_full=tplist, labels=tplist[:,0]) 
    
    return(tpunique, inverse_idx)
    
    