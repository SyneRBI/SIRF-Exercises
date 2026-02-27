class MAPRL(alg.Algorithm):

    def __init__(self, initial_estimate, data_fidelity, prior, step_size=1, relaxation_eta=0.01, eps=1e-8, **kwargs):

        self.initial_estimate = initial_estimate
        self.initial_step_size = step_size
        self.relaxation_eta = relaxation_eta
        self.eps = eps

        self.x = initial_estimate.clone()
        self.data_fidelity = data_fidelity
        self.prior = prior

        self.FOV_filter = pet.TruncateToCylinderProcessor()

        super(MAPRL, self).__init__(**kwargs)
        self.configured = True


    def step_size(self):
        return self.initial_step_size / (1 + self.relaxation_eta * self.iteration)
    
    def update(self):
        
        grad = self.data_fidelity.gradient(self.x) + self.prior.gradient(self.x)
        self.x = self.x - (self.x + self.eps) * grad * self.step_size()
        self.FOV_filter.apply(self.x)
        self.x.maximum(0, out=self.x)

    def update_objective(self):
        self.loss.append(self.data_fidelity(self.x) + self.prior(self.x))