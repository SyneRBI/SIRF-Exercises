class UnrolledOSEMVarNet(torch.nn.Module):
    def __init__(
        self,
        objective_function,
        sirf_template_image: sirf.STIR.ImageData,
        convnet: torch.nn.Module,
        device: str,
    ) -> None:
        """Unrolled OSEM Variational Network with 2 blocks

        Parameters
        ----------
        objective_function : sirf.STIR objetive function
            (listmode) Poisson logL objective function
            that we use for the OSEM updates
        sirf_template_image : sirf.STIR.ImageData
            used for the conversion between torch tensors and sirf images
        convnet : torch.nn.Module
            a (convolutional) neural network that maps a minibatch tensor 
            of shape [1,1,spatial_dimensions] onto a minibatch tensor of the same shape
        device : str
            device used for the calculations
        """
        super().__init__()

        # OSEM update layer using the 1st subset of the listmode data
        self._osem_step_layer0 = OSEMUpdateLayer(
            objective_function, sirf_template_image, 0, device
        )

        # OSEM update layer using the 2nd subset of the listmode data
        self._osem_step_layer1 = OSEMUpdateLayer(
            objective_function, sirf_template_image, 1, device
        )
        self._convnet = convnet
        self._relu = torch.nn.ReLU()

        # trainable parameters for the fusion of the OSEM update and the CNN output in the two blocks
        self._fusion_weight0 = torch.nn.Parameter(
            15 * torch.ones(1, device=device, dtype=torch.float32)
        )
        self._fusion_weight1 = torch.nn.Parameter(
            15 * torch.ones(1, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of the Unrolled OSEM Variational Network

        Parameters
        ----------
        x : torch.Tensor
            minibatch tensor of shape [1,1,spatial_dimensions] containing the image

        Returns
        -------
        torch.Tensor
            minibatch tensor of shape [1,1,spatial_dimensions] containing the output of the network
        """
        x1 = self._relu(
            self._fusion_weight0 * self._convnet(x) + self._osem_step_layer0(x)
        )
        x2 = self._relu(
            self._fusion_weight1 * self._convnet(x1) + self._osem_step_layer1(x1)
        )

        return x2
