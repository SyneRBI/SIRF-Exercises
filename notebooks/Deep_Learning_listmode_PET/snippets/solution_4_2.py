class OSEMUpdateLayer(torch.nn.Module):
    def __init__(self, objective_function, sirf_template_image: sirf.STIR.ImageData, subset: int, device: str) -> None:
        """OSEM update layer

        Parameters
        ----------
        objective_function : sirf (listmode) objective function
            the objective function that we use to calculate the gradient
        sirf_template_image : sirf.STIR.ImageData
            image template that we use to convert between torch tensors and sirf images
        subset : int
            subset number used for the gradient calculation
        device : str
            device used for the calculations

        Returns
        -------
        torch.Tensor
            minibatch tensor of shape [1,1,spatial_dimensions] containing the OSEM
            update of the input image using the Poisson log likelihood objective function
        """        
        super().__init__()
        self._objective_function = objective_function
        self._sirf_template_image: sirf.STIR.ImageData = sirf_template_image
        self._subset: int = subset

        self._poisson_logL_grad_layer = SIRFPoissonlogLGradLayer.apply

        # setup a tensor containng the inverse of the subset sensitivity image adding the minibatch and channel dimensions
        self._inv_sens_image: torch.Tensor = 1. / torch.tensor(objective_function.get_subset_sensitivity(subset).as_array(), dtype = torch.float32, device = device).unsqueeze(0).unsqueeze(0)
        # replace positive infinity values with 0 (voxels with 0 sensitivity)
        torch.nan_to_num(self._inv_sens_image, posinf = 0, out = self._inv_sens_image)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of the OSEM update layer

        Parameters
        ----------
        x : torch.Tensor
            minibatch tensor of shape [1,1,spatial_dimensions] containing the image

        Returns
        -------
        torch.Tensor
            OSEM update image
        """        
        grad_x: torch.Tensor = self._poisson_logL_grad_layer(x, self._objective_function, self._sirf_template_image, self._subset)
        return x + x * self._inv_sens_image * grad_x

