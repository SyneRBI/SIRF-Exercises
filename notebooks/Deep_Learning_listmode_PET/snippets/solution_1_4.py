lm_obj_fun.set_up(initial_image)

# initialize the reconstruction with ones where the sensitivity image is greater than 0
# all other values are set to zero and are not updated during reconstruction
lm_recon = initial_image.copy()
lm_recon.fill(lm_obj_fun.get_subset_sensitivity(0).as_array() > 0)

lm_inverse_sens_images = []

for i in range(num_subsets):
    lm_inverse_sens_image = acq_data.create_uniform_image(value=0, xy=nxny)
    lm_inverse_sens_image_np = np.zeros(
        lm_inverse_sens_image.shape, dtype=lm_inverse_sens_image.as_array().dtype
    )
    
    lm_sens_image_np = lm_obj_fun.get_subset_sensitivity(i).as_array()

    np.divide(
        1, lm_sens_image_np, out=lm_inverse_sens_image_np, where=lm_sens_image_np > 0
    )
    lm_inverse_sens_image.fill(lm_inverse_sens_image_np)
    lm_inverse_sens_images.append(lm_inverse_sens_image)

for it in range(num_iter):
    for i in range(num_subsets):
        subset_grad = lm_obj_fun.gradient(lm_recon, i)
        lm_recon = lm_recon + lm_recon * lm_inverse_sens_images[i] * subset_grad


fig4, ax4 = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
ax4.imshow(lm_recon.as_array()[71, :, :], cmap="Greys", vmin=0, vmax=vmax)
fig4.show()
