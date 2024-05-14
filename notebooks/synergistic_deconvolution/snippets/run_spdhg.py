spdhg_f1 = fn.KullbackLeibler(b=image_dict['OSEM'], eta=image_dict['OSEM'].get_uniform_copy(1e-6))
spdhg_f2 = fn.KullbackLeibler(b=image_dict['OSEM_amyloid'], eta=image_dict['OSEM_amyloid'].get_uniform_copy(1e-6))
spdhg_f3 = alpha * fn.MixedL21Norm()
spdhg_F = fn.BlockFunction(spdhg_f1, spdhg_f2, spdhg_f3)

spdhg_G = BlockIndicator(0)

grad = op.GradientOperator(image_dict['OSEM'])

k1 = op.BlockOperator(convolve, zero_operator_im2im, shape = (1,2))
k2 = op.BlockOperator(zero_operator_im2im, convolve, shape = (1,2))
joint_dgrad = op.BlockOperator(operator, zero_operator_im2grad, zero_operator_im2grad, operator, shape = (2,2))
k3 = op.CompositionOperator(MergeBlockDataContainers(joint_grad.range_geometry(),[1/6,5/6]), joint_dgrad)

spdhg_K = op.BlockOperator(k1, k2, k3)

# Define the step sizes sigma and tau (these are by no means optimal)
g = 0.1
sigma = [g*0.99/k.norm() for k in spdhg_K]
tau = min([0.99/k.norm()/g for k in spdhg_K])

spdhg = alg.SPDHG(f = spdhg_F, g = spdhg_G, operator = spdhg_K, tau = tau, sigma = sigma, initial=initial_estimate, update_objective_interval = 10, )

spdhg.run(verbose=2, iterations=500)