from pathlib import Path 
import numpy as np
import xmltodict
import scipy.signal as scisig
import nibabel as nib

from sirf.Utilities import assert_validity

import sirf.Reg as pReg
import sirf.Gadgetron as pMR
import sirf.DynamicSimulation as pDS

import matplotlib.pyplot as plt

def write_nii(fname, img):

	assert_validity(img, pMR.ImageData)

	nii = nib.Nifti1Image(np.squeeze(np.abs(img.as_array())), np.eye(4))
	nib.save(nii, fname)


def plot_array(arr):
	
	textcolor = 'white'
	assert arr.size == 3, "Please only pass 3D arrays"
	
	slcx,slcy,slcz = np.array(arr.shape)//2

	f, axs = plt.subplots(1,3)
	axs[0].imshow(arr[:,:,slcz])
	axs[0].set_ylabel("L-R")
	axs[0].set_xlabel("P-A")
	axs[0].set_xticks([])
	axs[0].set_yticks([])
	axs[0].xaxis.label.set_color(textcolor)
	axs[0].yaxis.label.set_color(textcolor)

	axs[1].imshow(arr[:,slcy,:])
	axs[1].set_ylabel("L-R")
	axs[1].set_xlabel("S-I")
	axs[1].set_xticks([])
	axs[1].set_yticks([])
	axs[1].xaxis.label.set_color(textcolor)
	axs[1].yaxis.label.set_color(textcolor)


	axs[2].imshow(arr[slcx,:,:])
	axs[2].set_ylabel("P-A")
	axs[2].set_xlabel("S-I")
	axs[2].set_xticks([])
	axs[2].set_yticks([])
	axs[2].xaxis.label.set_color(textcolor)
	axs[2].yaxis.label.set_color(textcolor)

	plt.show()


def read_motionfields(fpath_prefix):
	p = sorted( Path(fpath_prefix).glob('mvf*') )
	files = [x for x in p if x.is_file()]
	
	temp = []
	for f in files:
		print("Reading from {} ... ".format(f))
		img = pReg.NiftiImageData3DDisplacement(str(f))
		temp.append(img)

	data = np.array(temp, dtype=object)
	return data

def deform_segmentation(fname_segmentation, fpath_mvf, prefix_output_without_extension):
	# as a crosscheck you can store the motion in 3D and look at it and see if it needs tweaking
	resp_mvfs = read_motionfields(fpath_mvf)

	resampler = pReg.NiftyResampler()
	resampler.set_padding_value(0)
	resampler.set_interpolation_type_to_nearest_neighbour()

	segmentation = pReg.NiftiImageData3D(fname_segmentation)
	resampler.set_floating_image(segmentation)
	resampler.set_reference_image(segmentation)

	ci=0
	for mvf in resp_mvfs:
		print("Generating motionstate {}.".format(ci))
		resampler.clear_transformations()
		resampler.add_transformation(mvf)
		resampler.process()

		resampled_img = resampler.get_output()
		resampled_img = pReg.NiftiImageData3D(resampled_img.abs())
		tmp_name_output = prefix_output_without_extension + "{}.nii".format(ci)
		resampled_img.write(tmp_name_output)

		ci += 1


def coilmaps_from_rawdata(ad):

	assert_validity(ad, pMR.AcquisitionData)

	csm = pMR.CoilSensitivityData()
	csm.smoothness = 50
	csm.calculate(ad)

	return csm

def unity_coilmaps_from_rawdata(ad):
	
	csm = coilmaps_from_rawdata(ad)
	
	
	csm_datatype = csm.as_array().dtype
	csm_shape = csm.as_array().shape
	unity_csm = np.ones(shape=csm_shape, dtype=csm_datatype)
	csm.fill(unity_csm)

	return csm

def gaussian_2D_coilmaps(ad):

    csm = pMR.CoilSensitivityData()
    csm.smoothness = 1
    csm.calculate(ad)

    csm_arr = csm.as_array()
    img_size = csm_arr.shape[2:]

    assert len(img_size) == 2, "Only ask for a 2D coilmap please."

    X,Y = np.meshgrid(np.linspace(-img_size[0]/2,img_size[0]/2, img_size[0]), np.linspace(-img_size[1]/2,img_size[1]/2, img_size[1]))

    num_coils = csm_arr.shape[0]
    # put the centers fo gaussian coil profiles in a circle around the image center 
    coil_center_rad = img_size[0]/3
    coil_centers = [coil_center_rad*np.array([np.cos(2*np.pi*i/num_coils), np.sin(2*np.pi*i/num_coils)]) for i in range(num_coils)]

    csm_gauss = np.zeros(csm_arr.shape)
    # fix some arbitary width
    coilmap_width_pix = np.max(img_size)
    for ic in range(num_coils):
        csm_gauss[ic, ...] = np.exp( -((X-coil_centers[ic][0])**2 + (Y-coil_centers[ic][1])**2) / coilmap_width_pix**2)

    # do an SVD to extrac the principal components to avoid normalisation problems
    csm_flat = np.reshape(csm_gauss, (csm_gauss.shape[0], -1))
    __,__,V = np.linalg.svd(csm_flat, full_matrices=False)

	

    # csm_flat = np.reshape(V, csm_gauss.shape)
    csm_flat = np.reshape(csm_flat, csm_gauss.shape)

    csm_norm = np.sum( np.conj(csm_flat) * csm_flat,axis=0)[np.newaxis,...]

    csm_flat = csm_flat / csm_norm
    csm_flat = np.swapaxes(csm_flat,0,1)
    csm.fill(csm_flat.astype(np.complex64))

    return csm


def reconstruct_data(ad, csm=None):
	assert_validity(ad, pMR.AcquisitionData)
	if csm is not None:
		assert_validity(csm, pMR.CoilSensitivityData)
	else:
		csm = coilmaps_from_rawdata(ad)
	img = pMR.ImageData()
	img.from_acquisition_data(ad)

	am = pMR.AcquisitionModel(ad, img)
	am.set_coil_sensitivity_maps(csm)

	return am.inverse(ad)
 
def conjGrad(A,x,b,tol=0,N=10):

    r = b - A(x)
    p = r.copy()
    for i in range(N):
        Ap = A(p)
        alpha = np.vdot(p.as_array()[:],r.as_array()[:])/np.vdot(p.as_array()[:],Ap.as_array()[:])
        x = x + alpha*p
        r = b - A(x)
        if np.sqrt( np.vdot(r.as_array(), r.as_array()) ) < tol:
            print('Itr:', i)
            break
        else:
            beta = -np.vdot(r.as_array()[:],Ap.as_array()[:])/np.vdot(p.as_array()[:],Ap.as_array()[:])
            p = r + beta*p
    return x 


class EncOp:

	def __init__(self,am):
		assert_validity(am, pMR.AcquisitionModel)
		self._am = am

	def __call__(self,x):
		assert_validity(x, pMR.ImageData)
		return self._am.backward(self._am.forward(x))

def iterative_reconstruct_data(ad, csm=None, num_iter=10):
	
	assert_validity(ad, pMR.AcquisitionData)
	if csm is not None:
		assert_validity(csm, pMR.CoilSensitivityData)
	else:
		csm = coilmaps_from_rawdata(ad)
	
	img = pMR.ImageData()
	img.from_acquisition_data(ad)

	am = pMR.AcquisitionModel(ad, img)
	am.set_coil_sensitivity_maps(csm)
	E = EncOp(am)
	x0 = am.backward(ad)
	return conjGrad(E,x0,x0, tol=0, N=num_iter)

def gate_acquisition_data(ad, idx_corr, keep_bins):
    assert_validity(ad, pMR.AcquisitionData)
    assert np.max(keep_bins) < len(idx_corr), "You ask for bins ({})that have not been defined in idx_corr of length {}.".format(np.max(keep_bins), len(idx_corr))

    keep_idx = []

    for bin in keep_bins:
        keep_idx = np.append(keep_idx, sorted(idx_corr[bin]))

    keep_idx = np.unique(keep_idx)
    if not np.size(keep_idx):
        raise AssertionError("The keep index is empty. Gating removed all acquisitions")

    return ad.get_subset(keep_idx)

def prep_time_resolved_recon(ad,num_time_points):
	assert_validity(ad, pMR.AcquisitionData)
	# first compute the CSM based on all data
	csm = pMR.CoilSensitivityData()
	csm.smoothness = 50
	csm.calculate(ad)
	csm_arr = csm.as_array()

	# then activate the time-resolved reconstruction in repetition dimension
	ad = activate_timeresolved_reconstruction(ad, num_time_points)

		# this step sets up a time-resolved coilmap. The reconstruction checks if for each
	# time point a coilmap is present. 
	csm.calculate(ad) 
	# But we want to use the coilmap that was computed from the entire dataset
	# so we give every repetition the same coilmap
	num_reps = csm.as_array().shape[1]
	csm_arr = np.tile(csm_arr, (1,num_reps,1,1))
	csm_arr = np.swapaxes(csm_arr, 0, 1)# unfortunately these two axes have to be swapped.
	csm = csm.fill(csm_arr.astype(csm.as_array().dtype))

	return ad, csm

def reconstruct_timeresolved(ad, num_time_points):
    
	ad, csm = prep_time_resolved_recon(ad,num_time_points)
	recon = reconstruct_data(ad, csm)

	return recon, ad

def reconstruct_timeresolved_gated(ad, num_time_points, idx_corr, keep_bins):
	
	ad, csm = prep_time_resolved_recon(ad,num_time_points)
	ad = gate_acquisition_data(ad,idx_corr,keep_bins)
	recon = reconstruct_data(ad, csm)

	return recon, ad

def iterative_reconstruct_timeresolved(ad, num_time_points, num_cg_iterations):
    
	ad, csm = prep_time_resolved_recon(ad,num_time_points)
	recon = iterative_reconstruct_data(ad, csm, num_cg_iterations)

	return recon, ad

def get_normed_sinus_signal(t0_s, tmax_s, Nt, f_Hz):

	t_s = np.linspace(t0_s, tmax_s, Nt)
	sig = 0.5 * (1 + np.sin( 2*np.pi*f_Hz*t_s))
	return t_s, sig

def get_normed_sawtooth_signal(t0_s, tmax_s, Nt, f_Hz):

	t_s = np.linspace(t0_s, tmax_s, Nt)
	sig = 0.5*(1 + scisig.sawtooth(2*np.pi*f_Hz*t_s))
	return t_s, sig


	
def set_motionfields_from_path(modyn, fpath_prefix):

	assert_validity(modyn, pDS.MRMotionDynamic)
	mvfs = read_motionfields(fpath_prefix)

	for m in mvfs:
		modyn.add_displacement_field(m)



## mrf matching
# from PTBPyRecon

def match_dict(dict_sig, dict_theta, im_sig_1d, magnitude = False):

	output = np.zeros((im_sig_1d.shape[0], dict_theta.shape[1]), dtype=im_sig_1d.dtype)
	print(output.shape)

	num_subranges = 1

	try:

		subranges = np.arange(im_sig_1d.shape)
		subranges = np.array_split(subranges, num_subranges)
		print(subranges)

		for sr in subranges:
			dot_prod = np.dot(np.conj(dict_sig), im_sig_1d[:,sr].transpose())

	except MemoryError:
		print("Memory error, we will split the task into more sets.")
		num_subranges += 1

	return False


def match_dict_1d(dict_sig, dict_theta, im_sig_1d, magnitude = False):

    # Calculate dot-product between signal and dictionary
    if magnitude:
        dot_prod = np.dot(np.abs(dict_sig), np.abs(im_sig_1d.transpose()))
    else:
        dot_prod = np.dot(np.conj(dict_sig), im_sig_1d.transpose())

    # Find maximum
    idx = np.nanargmax(np.abs(dot_prod), axis=0)

    # Get T1 and T2
    match_theta = dict_theta[idx, :]

    # Get Rho and iRho
    dot_prod = np.sum(np.multiply(dict_sig[idx,:], im_sig_1d), axis=1)
    match_theta[:, 0] = np.real(dot_prod)
    match_theta[:, 3] = np.imag(dot_prod)

    return (match_theta, dict_sig[idx, :]*(match_theta[:,0] + 1j*match_theta[:,3])[:,np.newaxis])


# python functions for ismrmrd header modification

# a quick parse of the header into a dictionary
# allows us to modify it quickly such that the reconstruction
# can pick up that we want the reconstruction to be time-resolved

def set_reconSpace_matrixSize(ad, matrixSize):
	
	assert_validity(ad, pMR.AcquisitionData)

	hdr = ad.get_header()
	doc = xmltodict.parse(hdr)
	
	old_size_x = doc['ismrmrdHeader']['encoding']['reconSpace']['matrixSize']['x']
	old_size_y = doc['ismrmrdHeader']['encoding']['reconSpace']['matrixSize']['y']
	old_size_z = doc['ismrmrdHeader']['encoding']['reconSpace']['matrixSize']['z']

	doc['ismrmrdHeader']['encoding']['reconSpace']['matrixSize']['x'] = matrixSize[0]
	doc['ismrmrdHeader']['encoding']['reconSpace']['matrixSize']['y'] = matrixSize[1]
	doc['ismrmrdHeader']['encoding']['reconSpace']['matrixSize']['z'] = matrixSize[2]

	doc['ismrmrdHeader']['encoding']['reconSpace']['fieldOfView_mm']['x'] *= matrixSize[0]/old_size_x
	doc['ismrmrdHeader']['encoding']['reconSpace']['fieldOfView_mm']['y'] *= matrixSize[1]/old_size_y
	doc['ismrmrdHeader']['encoding']['reconSpace']['fieldOfView_mm']['z'] *= matrixSize[2]/old_size_z

	hdr = xmltodict.unparse(doc)
	ad.set_header(hdr)

	return ad


def set_encodingLimits_repetition(ad, num_recon_imgs):

	assert_validity(ad, pMR.AcquisitionData)

	hdr = ad.get_header()
	doc = xmltodict.parse(hdr)

	doc['ismrmrdHeader']['encoding']['encodingLimits']['repetition']['minimum'] = 0
	doc['ismrmrdHeader']['encoding']['encodingLimits']['repetition']['center'] = 0
	doc['ismrmrdHeader']['encoding']['encodingLimits']['repetition']['maximum'] = num_recon_imgs

	hdr = xmltodict.unparse(doc)
	ad.set_header(hdr)

	return ad

def activate_timeresolved_reconstruction(ad, num_recon_imgs):

	assert_validity(ad, pMR.AcquisitionData)
	set_encodingLimits_repetition(ad, num_recon_imgs)

	ad_resolved = ad.new_acquisition_data()

	# this way we will reconstruct one image per readout
	for ia in range(ad.number()):
		acq = ad.acquisition(ia)
		acq.set_repetition(int(np.floor(ia / ad.number()* num_recon_imgs)))
		ad_resolved.append_acquisition(acq)

	ad_resolved.sort_by_time()

	return ad_resolved

def apply_databased_sliding_window(ad, data):
    
    assert_validity(ad, pMR.AcquisitionData)
	
    assert data.shape[0] >= ad.number(), "Please give a dataset with the same data size in the 0th dimension as there are acquisitions."

    repetition_number = np.array(ad.get_ISMRMRD_info('repetition'), dtype=np.intc)
    unique_reps = np.unique(repetition_number)
    repetition_number.shape[0]
    
    avg_data = np.zeros(shape=(len(unique_reps), data.shape[1]), dtype=data.dtype)

    for ur in range(unique_reps.size):
        avg_data[ur,:] = np.mean( data[np.where( repetition_number==unique_reps[ur])[0],...], axis=0)
    
    return avg_data


def gate_data(data, idx_corr, keep_bins):
    
    keep_idx = []

    for bin in keep_bins:
        keep_idx = np.append(keep_idx, sorted(idx_corr[bin]))

    keep_idx = np.array(np.unique(keep_idx), dtype=int)
    if not np.size(keep_idx):
        raise AssertionError("The keep index is empty. Gating removed all data.")
     
    rd = data[keep_idx,...]
    
    return rd

# match dictionary 
def match_dict_1d(dict_sig, dict_theta, im_sig_1d, magnitude = False):

    # Calculate dot-product between signal and dictionary
    if magnitude:
        dot_prod = np.dot(np.abs(dict_sig), np.abs(im_sig_1d.transpose()))
    else:
        dot_prod = np.dot(np.conj(dict_sig), im_sig_1d.transpose())

    # Find maximum
    idx = np.nanargmax(np.abs(dot_prod), axis=0)

    # Get T1 and T2
    match_theta = dict_theta[idx, :]

    # Get Rho and iRho
    dot_prod = np.sum(np.multiply(dict_sig[idx,:], im_sig_1d), axis=1)
    match_theta[:, 0] = np.real(dot_prod)
    match_theta[:, 3] = np.imag(dot_prod)

    return (match_theta, dict_sig[idx, :]*(match_theta[:,0] + 1j*match_theta[:,3])[:,np.newaxis])


# do pixel-wise matching on lower memory by splitting it up 
def match_dict(dict_sig, dict_theta, im_sig_1d, magnitude = False):

    output = np.zeros((im_sig_1d.shape[0], dict_theta.shape[1]), dtype=im_sig_1d.dtype)

    num_subsets = 32
    success = False
    while not success:
        try:
            subranges = np.arange(im_sig_1d.shape[0])
            subranges = np.array_split(subranges, num_subsets)

            for sr in subranges:
                match_subrange = match_dict_1d(dict_sig, dict_theta, im_sig_1d[sr,:])
                output[sr,:] = match_subrange[0]
                
            success = True

        except MemoryError:
            num_subsets *= 2
            print("Memory error, we will split the task into {} sets.".format(num_subsets))

    return output


def dictionary_matching(recon, dict_sig, dict_theta):

	assert_validity(recon, pMR.ImageData)
	
	img_series = recon.as_array()
	img_shape = img_series.shape[1:]
	img_series_1d = np.transpose(np.reshape(img_series,(img_series.shape[0], -1)))
	
	# this checks the largest overlap between time-profile and dictionary entries
	# if the RAM overflows this will catch it and perform the task in multiple sets.
	dict_match = match_dict(dict_sig, dict_theta, img_series_1d)

	dict_match = np.reshape(dict_match, (*img_shape, -1))
	return dict_match