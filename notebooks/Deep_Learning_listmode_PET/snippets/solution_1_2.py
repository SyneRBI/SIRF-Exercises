subset = 0
subset_grad = obj_fun.gradient(initial_image, subset)
# this is only correct, if the sensitivity image is greater than 0 everywhere
# (see next exercise for more details)
step = initial_image / obj_fun.get_subset_sensitivity(subset)
osem_update = initial_image + step * subset_grad

# maximum value of the updated image is nan, because the sensitivity image is 0 in some places
# which needs special attention
print(osem_update.max())
