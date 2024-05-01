# %% [markdown]
# Learning objectives
# ===================
#
# 1. Basics of pytorch's automatic gradient calculation (autograd)
# 2. How to implement the "backward" pass for a layer that perform a
#    gradient ascent step according the the PET objective function (data fidelity)
# 3. How to test whether the backward pass is implemented correctly (torch.autograd.gradcheck).   