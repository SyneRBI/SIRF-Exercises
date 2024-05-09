lm_obj_fun.set_up(initial_image)

# initialize the reconstruction with ones where the sensitivity image is greater than 0
# all other values are set to zero and are not updated during reconstruction
lm_recon = initial_image.copy()
lm_recon.fill(obj_fun.get_subset_sensitivity(0).as_array() > 0)

# setup an image to store the step size
step = acq_data.create_uniform_image(value=1, xy=nxny)

for it in range(num_iter):
    for i in range(num_subsets):
        subset_grad = lm_obj_fun.gradient(recon, i)
        tmp = np.zeros(recon.shape, dtype=recon.as_array().dtype)
        # use np.divide for the element-wise division for all elements where
        # the sensitivity image is greater than 0
        np.divide(
            recon.as_array(),
            obj_fun.get_subset_sensitivity(
                i
            ).as_array(),  ### HACK: lm_obj_fun.get_subset_sensitivity(i) still returns None
            out=tmp,
            where=obj_fun.get_subset_sensitivity(i).as_array() > 0,
        )
        step.fill(tmp)
        lm_recon = lm_recon + step * subset_grad

fig4, ax4 = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
ax4.imshow(lm_recon.as_array()[71, :, :], cmap="Greys", vmin=0, vmax=vmax)
fig4.show()
