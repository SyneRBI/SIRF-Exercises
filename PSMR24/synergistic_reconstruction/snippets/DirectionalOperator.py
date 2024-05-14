class DirectionalOperator(op.LinearOperator):

    def __init__(self, anatomical_gradient, gamma = 1, eta=1e-6):

        self.anatomical_gradient = anatomical_gradient
        geometry = cil.BlockGeometry(*[x for x in anatomical_gradient.containers])
        self.tmp = self.anatomical_gradient.containers[0].clone()

        self.gamma = gamma

        self.xi = self.anatomical_gradient/(self.anatomical_gradient.pnorm().power(2)+eta**2).sqrt()

        super(DirectionalOperator, self).__init__(domain_geometry=geometry,
                                       range_geometry=geometry,)
        
    def direct(self, x, out=None):

        if out is None:
            return x - self.gamma * self.xi * self.dot(self.xi, x)
        else:
            out.fill(x - self.gamma * self.xi * self.dot(self.xi, x))
    
    def adjoint(self, x, out=None):
        # This is the same as the direct operator
        return self.direct(x, out)
    
    def dot(self, x, y):
        self.tmp.fill(0)
        for el_x, el_y in zip(x.containers, y.containers):
            self.tmp += el_x * el_y
        return self.tmp
    

grad_ref = grad.direct(image_dict['T1'])
d_op = DirectionalOperator(grad_ref)

def calculate_norm(self):
    # return product of operator norms
    return np.prod([op.norm() for op in self.operators])
op.CompositionOperator.calculate_norm = calculate_norm

operator = op.CompositionOperator(d_op, grad)

Kd = op.BlockOperator(convolve, operator)