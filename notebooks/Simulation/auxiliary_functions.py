from pathlib import Path 
import numpy as np
import xmltodict


from sirf.Utilities import assert_validity

import sirf.Reg as pReg
import sirf.Gadgetron as pMR
import sirf.DynamicSimulation as pDS



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

	

def get_normed_surrogate_signal(t0_s, tmax_s, Nt, f_Hz):

	t_s = np.linspace(t0_s, tmax_s, Nt)
	sig = 0.5 * (1 + np.sin( 2*np.pi*f_Hz*t_s))
	return t_s, sig

	
def set_motionfields_from_path(modyn, fpath_prefix):

	assert_validity(modyn, pDS.MRMotionDynamic)
	mvfs = read_motionfields(fpath_prefix)

	for m in mvfs:
		modyn.add_displacement_field(m)



## mrf matching
# from PTBPyRecon

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
	set_encodingLimits_repetition(ad_resolved, num_recon_imgs)

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
	
    assert data.shape[0] == ad.number(), "Please give a dataset with the same data size in the 0th dimension as there are acquisitions."

    repetition_number = np.array(ad.get_ISMRMRD_info('repetition'), dtype=np.intc)
    unique_reps = np.unique(repetition_number)
    repetition_number.shape[0]
    
    avg_data = np.zeros(shape=(len(unique_reps), data.shape[1]))

    for ur in unique_reps:
        avg_data[ur,:] = np.mean( data[np.where( repetition_number==ur)[0],...], axis=0)
    
    return avg_data

# use CG solver for quick iterative reconstruction

