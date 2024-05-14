grad_ref = grad.direct(image_dict['T1'])
d_op = DirectionalOperator(grad_ref)

def calculate_norm(self):
    # return product of operator norms
    return np.prod([op.norm() for op in self.operators])
op.CompositionOperator.calculate_norm = calculate_norm

operator = op.CompositionOperator(d_op, grad)

Kd = op.BlockOperator(convolve, operator)

# Define the step sizes sigma and tau (these are by no means optimal)
normK = K.norm()
sigma = 1./normK
tau = 1./normK

pdhg_dtv = alg.PDHG(f = f, g = g, operator = Kd, tau = 0.99*tau, sigma = 0.99*sigma, initial=image_dict['OSEM'],
                    update_objective_interval = 50, check_convergence=False)

pdhg_dtv.run(verbose=1, iterations=500)

