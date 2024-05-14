alpha = 0.02
f1 = fn.KullbackLeibler(b=image_dict['OSEM'], eta=image_dict['OSEM'].get_uniform_copy(1e-6))
f2 = alpha * fn.MixedL21Norm()
f = fn.BlockFunction(f1, f2)

g = fn.IndicatorBox(0)

grad = op.GradientOperator(image_dict['OSEM'])
K = op.BlockOperator(convolve, grad)

# Define the step sizes sigma and tau (these are by no means optimal)
normK = K.norm()
sigma = 1./normK
tau = 1./normK

pdhg = alg.PDHG(f = f, g = g, operator = K, tau = 0.99*tau, sigma = 0.99*sigma, initial=image_dict['OSEM'], 
                update_objective_interval = 50, check_convergence=False)

pdhg.run(verbose=1, iterations=500)
