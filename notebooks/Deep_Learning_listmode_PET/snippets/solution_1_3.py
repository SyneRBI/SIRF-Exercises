# calculate the inverse of the subset sensitivity images, correctly accounting for voxels
# where the sensitivity images are zero

inverse_sens_images = []

for i in range(num_subsets):
    inverse_sens_image = acq_data.create_uniform_image(value=0, xy=nxny)
    inverse_sens_image_np = np.zeros(
        inverse_sens_image.shape, dtype=inverse_sens_image.as_array().dtype
    )
    sens_image_np = obj_fun.get_subset_sensitivity(i).as_array()
    np.divide(1, sens_image_np, out=inverse_sens_image_np, where=sens_image_np > 0)
    inverse_sens_image.fill(inverse_sens_image_np)
    inverse_sens_images.append(inverse_sens_image)

for it in range(num_iter):
    for i in range(num_subsets):
        subset_grad = obj_fun.gradient(recon, i)
        # use np.divide for the element-wise division for all elements where
        # the sensitivity image is greater than 0
        recon = recon + recon * inverse_sens_images[i] * subset_grad

fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
ax2.imshow(recon.as_array()[71, :, :], cmap="Greys", vmin=0, vmax=vmax)
fig2.show()
