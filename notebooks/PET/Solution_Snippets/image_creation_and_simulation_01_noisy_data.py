noisy_array=numpy.random.poisson(acquired_data_with_attn.as_array()).astype('float64')
print(' Maximum counts in the data: %d' % noisy_array.max())
# stuff into a new AcquisitionData object
noisy_data = acquired_data_with_attn.clone()
noisy_data.fill(noisy_array);

# Display bitmaps of the middle sinogram
acquired_data_with_attn.show()
noisy_data.show()