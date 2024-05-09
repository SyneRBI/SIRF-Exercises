# setup an image to store the step size
step = acq_data.create_uniform_image(value=1, xy=nxny)

for it in range(num_iter):
    for i in range(num_subsets):
        subset_grad = obj_fun.gradient(recon, i)
        tmp = np.zeros(recon.shape, dtype=recon.as_array().dtype)
        # use np.divide for the element-wise division for all elements where
        # the sensitivity image is greater than 0
        np.divide(
            recon.as_array(),
            obj_fun.get_subset_sensitivity(i).as_array(),
            out=tmp,
            where=obj_fun.get_subset_sensitivity(i).as_array() > 0,
        )
        step.fill(tmp)
        recon = recon + step * subset_grad

fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
ax2.imshow(recon.as_array()[71, :, :], cmap="Greys", vmin=0, vmax=vmax)
fig2.show()
