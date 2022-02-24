# %% [markdown]
# ## Notbook A: How to preprocess input data for simulation
# 
# #### Prerequisites:
# - basic knowledge of Python 
# - basic understanding of image orientation 
# 
# #### Goals:
# - preparing valid simulation input data.
# 
# #### Content overview: 
# - formatting an anatomy segmentation into Nifti
# - formatting displacement vector fields into Nifti
# - formatting orientation of MR rawdata
# - Writing XML rawdata input
# - using simulation to store ground truth T1, T2 and spin density maps

# %% [markdown]
# First off, we require to pre-process our data.
# We require the following minimum input data for a simulation:
# - an MR rawdata file in ISMRMRD format describing the segmentation (contrast template)
# - an MR rawdata file in ISMRMRD format describing the acquisition process (acquisition template)
# - an anatomy segmentation in RAI format (right-anterior-inferior)
# - an XML descriptor assigning tissue parameters to the labels.
# Optional:
# - motion information matching the anatomy segmentation
# - contrast information describing the signal over time

# %%

# first we generate where we store our data
import os
from pathlib import Path 

# our background is dark
textcolor = 'white'

# this is where our data is currently
root_path = '/media/sf_CCPPETMR/TestData/Input/xDynamicSimulation/pDynamicSimulation/'

# this is where we store the properly formatted data
fpath_out = Path(os.getenv("SIRF_INSTALL_PATH"))
fpath_out = fpath_out / "share/SIRF-3.1/Simulation/Input/"


fpath_out.mkdir(exist_ok=True,parents=True)

# %% [markdown]
# ### Raw data

# %% [markdown]
# The simulation needs two MR raw-data objects to serve as templates:
# - one to set up the segmentation in image space, the so-called contrast template
# - one to dictate the simulated acquisition, the so-called acquisition template
# 
# Since the simulation is able to resample different containers their geometry needs to be defined first. 
# This is done by using a pre-compiled executable usually used for the SIRF tests.
# 

# %%
fpath_SIRF_build = "/home/sirfuser/devel/buildVM/builds/SIRF/build/"
fpath_preprocess_exe = fpath_SIRF_build + "src/xGadgetron/cGadgetron/tests/MR_PROCESS_TESTDATA"

fname_input_contrast_template = root_path + 'Cube128/CV_nav_cart_128Cube_FLASH_T1.h5'
fname_contrast_template = str(fpath_out) + "/contrast_template.h5"
command_conttempl = "{} {} {}".format(fpath_preprocess_exe, fname_input_contrast_template, fname_contrast_template)


fname_input_acquisition_template = root_path + 'General/meas_MID33_rad_2d_gc_FID78808_ismrmrd.h5'
fname_acquisition_template = str(fpath_out) + "/acquisition_template.h5"
command_acqtemplate = "{} {} {}".format(fpath_preprocess_exe, fname_input_acquisition_template, fname_acquisition_template)


# %%
# run the external commands and catch an error
import subprocess 

try:
    subprocess.call(command_conttempl, shell=True)
except subprocess.CalledProcessError as e:
    print(e.output)

try:
    subprocess.call(command_acqtemplate, shell=True)

except subprocess.CalledProcessError as e:
    print(e.output)

# %% [markdown]
# ### Segmentation

# %%
import nibabel as nib

fpath_segmentation_nii = root_path + 'Cube128/label_volume_rai.nii'

segmentation = nib.load(fpath_segmentation_nii)
print("The data shape is {}".format(segmentation.shape))

# %% [markdown]
# Let's have a look at the data:

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np



def plot_array(arr):
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

plot_array(segmentation.get_fdata())


# %% [markdown]
# Ideally these data are already in RAI orientation.
# RAI means: the data are available in memory such with increasing XYZ index the voxels move from left to right, from posteior to anterior, and from superior to inferior.
# Now this needs to be stored with orientation as well as resolution information.

# %%
import sirf.Reg as pReg

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

fpath_resp_mvf = root_path + 'Cube128/mvf_resp/'
resp_mvfs = read_motionfields(fpath_resp_mvf)

# %%
# now we plot motion fields and segmentation and see if they overlap
inhale_mvf = np.squeeze(resp_mvfs[-1].as_array())
inhale_abs = np.linalg.norm(inhale_mvf,axis=-1)
plot_array(segmentation.get_fdata())
plot_array(inhale_abs)

# %%
# it seems our motion fields are in LPS.

# if we flip array indices of a DVF we must ensure the 
def flip_mvf(mvf, axis):

    mvf = np.flip(mvf, axis=axis)
    mvf[:,:,:,axis] *= -1
    
    return mvf

# from above we see we have to flip in all three directions
inhale_mvf = flip_mvf(inhale_mvf,axis=0)
inhale_mvf = flip_mvf(inhale_mvf,axis=1)
inhale_mvf = flip_mvf(inhale_mvf,axis=2)
inhale_abs = np.linalg.norm(inhale_mvf,axis=-1)

# check again if the motion fields match the anatomy now
plot_array(segmentation.get_fdata())
plot_array(inhale_abs)

# %%
# now we can write a reformat function for our motion vector fields:
def reformat_mvfs( mvfs ):
    out = []
    for cmvf in mvfs:
        cmvf = np.squeeze(cmvf.as_array())
        cmvf = flip_mvf(cmvf,axis=0)
        cmvf = flip_mvf(cmvf,axis=1)
        cmvf = flip_mvf(cmvf,axis=2)
        cmvf = cmvf[:,:,:,np.newaxis,:]
        out.append(cmvf)

    return np.array(out, dtype=object)

resp_mvfs = reformat_mvfs(resp_mvfs)

# %%
# so far we only saw some voxelised data. Now we need to store it with approriate geometry information
# such that the simulation knows where the voxels are.

# now we need to store segmentation and motion fields with geometry information.
# NIFTI expects an RAI coordidnate system.

resolution_mm_per_pixel = np.array([2,2,-2,1])

# the first voxel center is lies at -FOV / 2 + dx/2

offset_mm =(-np.array(segmentation.shape)/2 + 0.5) * resolution_mm_per_pixel[0:3]
affine = np.diag(resolution_mm_per_pixel)
affine[:3,3] = offset_mm


# %%

# store segmentation as nifti
img = nib.Nifti1Image(segmentation.get_fdata(), affine)
# this is crucial since otherwise the offset won't end up in the nifti
img.set_qform(affine)

fname_segmentation = str(fpath_out) + "/segmentation.nii"
nib.save(img, fname_segmentation)


# %%
# store motion fields as nifti
def store_mvfs(fpath_mvf_output, mvfs, affine):

    for i in range(mvfs.shape[0]):
        fname_mvf_output = fpath_mvf_output + "/mvf_{}".format(i)
        tmp = mvfs[i,:].astype(np.float32)
        img = nib.Nifti1Image(tmp, affine)

        print("Storing {}".format(fname_mvf_output))
        nib.save(img, fname_mvf_output)

fpath_mvf_output = fpath_out / 'mvfs_resp'
fpath_mvf_output.mkdir(exist_ok=True,parents=True)

store_mvfs(str(fpath_mvf_output), resp_mvfs, affine)
del resp_mvfs

# %%
# Same for cardiac motion fields
fpath_card_mvf = root_path + 'Cube128/mvf_card/'
card_mvfs = read_motionfields(fpath_card_mvf)
card_mvfs = reformat_mvfs(card_mvfs)

fpath_mvf_output = fpath_out / 'mvfs_card'
fpath_mvf_output.mkdir(exist_ok=True,parents=True)

store_mvfs(str(fpath_mvf_output), card_mvfs, affine)
del card_mvfs

# %% [markdown]
# ### XML descriptor 

# %% [markdown]
# Here is an example of the XML descriptor. 
# 
# It contains
# - the label | which voxels it describes in the segmentation
# - the name | keeps track of what we actually mean by it 
# - MR parameters | T1, T2, proton density and chemical shift
# - (PET parameters | only relevant for PET simulations)
# 
# <img src="./simulation_xml_example.png" alt="drawing" width=600 />
# 
# 
# 
# <h4> Features and Caveats: </h4>
# 
# - the XML descriptor **needs to come in the form as displayed**.
# - arbitarily many sections of TissueParameter objects can be added, it does not matter if there are parameters described that do not appear in your segmentation.
# - **all labels appearing in your segmentations need to appear**, otherwise an error will occur informing you of your mistake. The above example would require a couple of more entries.
# - the PET parameters are not taken into consideration for MR simulations, but they still need to be present, fill them with zeros.
# 

# %%

import sirf.Reg as pReg
import sirf.DynamicSimulation as pDS
import sirf.Gadgetron as pMR


fpath_input = fpath_out

fpath_out = Path(os.getenv("SIRF_INSTALL_PATH"))
fpath_out = fpath_out / "share/SIRF-3.1/Simulation/Output/"
fpath_out.mkdir(exist_ok=True,parents=True)

prefix_ground_truth = str(fpath_out) + "/simulation_geometry_acquisition_offset_parametermap"


# %%

# load the templates into SIRF MR acquisition data container
contrast_template = pMR.AcquisitionData(fname_contrast_template)
acquisition_template = pMR.AcquisitionData(fname_acquisition_template)

print(contrast_template.number())
print(acquisition_template.number())


# %%

# load the labels into SIRF Nifti Container
segmentation = pReg.NiftiImageData3D(fname_segmentation)

# set up the simulation with the segmentation and corresponding XML filename.

fname_xml = fpath_input / 'XCAT_TissueParameters_XML.xml'
if fname_xml.exists():
    mrsim = pDS.MRDynamicSimulation(segmentation, str(fname_xml))
else:
    raise AssertionError("You didn't provide the XML file {} we were looking for.".format(fname_xml))


# %%

mrsim.set_contrast_template_data(contrast_template)
mrsim.set_acquisition_template_data(acquisition_template)


# %%
# now we set up an affine transformation to move the acquired slice around
offset_x_mm = 0
offset_y_mm = 0
offset_z_mm = 0
rotation_angles_deg = [0,0,0]
translation = np.array([offset_x_mm, offset_y_mm, offset_z_mm])
euler_angles_deg = np.array(rotation_angles_deg)

offset_trafo = pReg.AffineTransformation(translation, euler_angles_deg)
mrsim.set_offset_trafo(offset_trafo)

filenames_parametermaps = mrsim.save_parametermap_ground_truth(prefix_ground_truth)

# %%
t1_map = pReg.NiftiImageData(filenames_parametermaps[0])
t2_map = pReg.NiftiImageData(filenames_parametermaps[1])
rho_map = pReg.NiftiImageData(filenames_parametermaps[2])

f, axs = plt.subplots(1,3)
axs[0].imshow(np.transpose(t1_map.as_array()))
axs[0].axis("off")
axs[0].set_title("T1")
axs[0].title.set_color(textcolor)

axs[1].imshow(np.transpose(t2_map.as_array()))
axs[1].axis("off")
axs[1].set_title("T2")
axs[1].title.set_color(textcolor)

axs[2].imshow(np.transpose(rho_map.as_array()))
axs[2].axis("off")
axs[2].set_title("Spin density")
axs[2].title.set_color(textcolor)

plt.show()
f.savefig('/media/sf_CCPPETMR/fig_gtmaps', dpi=300)


# %% [markdown]
# ### Recap
# In this notebook we:
# 
# - added geometry information to a voxelised antatomy segmentation,
# - ensured the motion information as displacement vector fields matched the anatomy,
# - reset our MR template rawdata to have the correct geometry stored,
# - learned about the XML file describing the tissue parameters,
# - added a transformation that leads to 2D images in 4 chamber view.
# 
# Congratulations you can continue to use the simulation framework to generate some simulations.

# %% [markdown]
# 


