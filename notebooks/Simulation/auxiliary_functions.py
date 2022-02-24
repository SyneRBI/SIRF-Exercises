from pathlib import Path 
import numpy as np


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


def reconstruct_data(ad, csm=None):
	assert_validity(ad, pMR.AcquisitionData)
	if csm is not None:
		assert_validity(csm, pMR.CoilSensitivityData)

	img = pMR.ImageData()
	img.from_acquisition_data(ad)

	csm = pMR.CoilSensitivityData()
	csm.smoothness = 50
	csm.calculate(ad)

	am = pMR.AcquisitionModel(ad, img)
	am.set_coil_sensitivity_maps(csm)

	return am.inverse(ad)


def get_normed_surrogate_signal(t0_s, tmax_s, Nt, f_Hz):

	t_s = np.linspace(t0_s, tmax_s, Nt)
	sig = 0.5 * (1 + np.sin( 2*np.pi*f_Hz*t_s))
	return t_s, sig

	
def set_motionfields_from_path(modyn, fpath_prefix):

	assert_validity(modyn, pDS.MRMotionDynamic)
	mvfs = read_motionfields(fpath_prefix)

	for m in mvfs:
		modyn.add_displacement_field(m)