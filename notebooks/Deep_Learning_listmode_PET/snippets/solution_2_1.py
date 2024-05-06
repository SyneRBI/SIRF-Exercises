# we first create an numpy array that we fill in a loop
numpy_image_3: np.ndarray = np.zeros(sirf_image_1.shape, dtype = sirf_image_1.dtype)
for i in range(numpy_image_3.shape[0]):
    numpy_image_3[i,:,:] = i**2

sirf_image_3: sirf.STIR.ImageData = acq_data.create_uniform_image(0.0)
sirf_image_3.fill(numpy_image_3)

print()
print(f"sirf_image_3 shape   .: {sirf_image_3.shape}")
print(f"sirf_image_3 spacing .: {sirf_image_3.spacing}")
print(f"sirf_image_3 max     .: {sirf_image_3.max()}")
