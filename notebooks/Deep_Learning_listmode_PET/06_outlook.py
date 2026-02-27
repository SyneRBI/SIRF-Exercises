# %% [markdown]
# Outlook
# =======
#
# Congratulations, you made through all the notebooks and should
# now be able to understand the basics of deep learning for (listmode) PET reconstruction.
#
# Follow-up questions / proposals to think about
# ----------------------------------------------
#
# 1. How could we incorporate training on mini-batches with a batch size greater than 1?
# 2. What happens if we increase the number of unrolled blocks in the variational network?
# 3. What is the impact on the CNN size / architecture and the loss function in the variational network?
# 4. Most neural networks work best if the image intensities are normalized. How could we include this in the training process?
#
# Training speed
# --------------
#
# Currently, the training is quite slow due to several reasons:
# - the listmode projections are currently performed on the CPU
# - there is a lot of memory transfer between CPU and GPU during training (OSEM block on CPU, CNN on GPU)
#
# **However,** SIRF and STIR are community projects that are constantly being improved.
# So, it is likely that the training speed will increase in the future.
