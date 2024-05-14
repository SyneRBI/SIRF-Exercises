class BlockIndicator(Function):
    def __init__(self, lower_bound=0, upper_bound=np.inf):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound        
        
    def __call__(self, x):
        if isinstance(x, cil.BlockDataContainer):
            # return inf if any element is outside the bounds
            for el in x.containers:
                if el.as_array().min() < self.lower_bound or el.as_array().max() > self.upper_bound:
                    return np.inf
            return 0
        else:
            if x.as_array().min() < self.lower_bound or x.as_array().max() > self.upper_bound:
                return np.inf
            return 0
        
    def proximal(self, x, tau, out=None):
        tmp = x.clone()
        tmp.maximum(self.lower_bound, out=tmp)
        tmp.minimum(self.upper_bound, out=tmp)
        if out is None:
            return tmp
        out.fill(tmp)
    
    def convex_conjugate(self, x):
        return x.maximum(0).sum()