## This file contains helper functions that are used in the notebook
## They ahould all be duplicated in the notebook for clarity
## But feel free to use these for further development of notebooks or scripts

import numpy as np
import sirf.STIR as pet

def fwhm_to_sigma(fwhm):
    ''' Converts full width at half maximum (FWHM) to standard deviation (sigma)'''
    return fwhm / (2 * np.sqrt(2 * np.log(2)))

def psf(n, fwhm, voxel_size=(1, 1, 1)):
    ''' Creates a 3D point spread function (PSF) with specified sizes `n`, FWHM values `fwhm`, and voxel sizes `voxel_size` '''

    # Convert FWHM to sigma and scale by voxel size
    sigma_voxels = [fwhm_to_sigma(fwhm[i]) / voxel_size[i] for i in range(3)]
    
    # Create Gaussian distributions for each dimension
    axes = [np.linspace(-(n - 1) / 2., (n - 1) / 2., n) for i in range(3)]
    gauss = [np.exp(-0.5 * np.square(ax) / np.square(sigma_voxels[i])) for i, ax in enumerate(axes)]

    # Create 3D Gaussian kernel
    kernel_3d = np.outer(gauss[0], gauss[1]).reshape(n, n, 1) * gauss[2].reshape(1, 1, n)
    
    # Normalize the kernel to ensure its sum equals one
    return kernel_3d / np.sum(kernel_3d)

def make_acquisition_model(template_sinogram, template_image, atten_image):
    ''' Creates an acquisition model object using the Ray Tracing Matrix method with the specified template sinogram, template image, and attenuation image '''

    # We'll start by initialising the acquisition model object
    acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
    acq_model.set_num_tangential_LORs(10) # sets the number of LORs for each bin in the sinogram

    # In order to create LOR attenuation coefficients, we need to project the attenuation image into sinogram space
    # And then create an acquisition sensitivity model using these attenuation coefficients
    acq_asm = pet.AcquisitionModelUsingRayTracingMatrix()
    acq_asm.set_num_tangential_LORs(5)
    acq_model.set_acquisition_sensitivity(pet.AcquisitionSensitivityModel(atten_image, acq_asm))

    # And finally, we can set up the acquisition model
    acq_model.set_up(template_sinogram,template_image)

    return acq_model

def add_poission_noise(acquistion_data, noise_level=10, seed=10):
    ''' Adds Poisson noise to the acquisition data with a specified noise level and seed '''

    np.random.seed(seed)
    
    # Divide by and then multiply by noise level to scale the amount of noise (of course this doesn't respect the Poisson noise model, but it's a simple way to add noise to the data)
    noisy_data = np.random.poisson(acquistion_data.as_array()/noise_level)*noise_level
    return acquistion_data.clone().fill(noisy_data)