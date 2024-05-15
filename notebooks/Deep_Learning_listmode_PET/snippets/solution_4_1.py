class SIRFPoissonlogLGradLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, objective_function, sirf_template_image: sirf.STIR.ImageData, subset: int) -> torch.Tensor:
        """(listmode) Poisson loglikelihood gradient layer forward pass

        Parameters
        ----------
        ctx : context object
            used to store objects that we need in the backward pass
        x : torch.Tensor
            minibatch tensor of shape [1,1,spatial_dimensions] containing the image
        objective_function : sirf (listmode) objective function
            the objective function that we use to calculate the gradient
        sirf_template_image : sirf.STIR.ImageData
            image template that we use to convert between torch tensors and sirf images
        subset : int
            subset number used for the gradient calculation

        Returns
        -------
        torch.Tensor
            minibatch tensor of shape [1,1,spatial_dimensions] containing the image
            containing the gradient of the (listmode) Poisson log likelihood at x
        """

        # we use the context object ctx to store the matrix and other variables that we need in the backward pass
        ctx.device = x.device
        ctx.objective_function = objective_function
        ctx.dtype = x.dtype 
        ctx.subset = subset
        ctx.sirf_template_image = sirf_template_image

        # setup a new sirf.STIR ImageData object 
        x_sirf = sirf_template_image.clone()
        # convert torch tensor to sirf image via numpy
        x_sirf.fill(x.cpu().numpy()[0,0,...]) 

        # save the input sirf.STIR ImageData for the backward pass
        ctx.x_sirf = x_sirf

        # calculate the gradient of the Poisson log likelihood using SIRF
        g_np = objective_function.gradient(x_sirf, subset).as_array()

        # convert back to torch tensor
        y = torch.tensor(g_np, device=ctx.device, dtype = ctx.dtype).unsqueeze(0).unsqueeze(0)

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor | None) -> tuple[torch.Tensor | None, None, None, None]:
        """(listmode) Poisson loglikelihood gradient layer backward pass

        Parameters
        ----------
        ctx : context object
            used to store objects that we need in the backward pass
        grad_output : torch.Tensor | None
            minibatch tensor of shape [1,1,spatial_dimensions] containing the gradient (called v in the autograd tutorial)
            https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#optional-reading-vector-calculus-using-autograd

        Returns
        -------
        tuple[torch.Tensor | None, None, None, None]
            the Jacobian-vector product of the Poisson log likelihood gradient layer
        """        

        if grad_output is None:
            return None, None, None, None
        else:
            # convert torch tensor to sirf image via numpy
            ctx.sirf_template_image.fill(grad_output.cpu().numpy()[0,0,...])

            # calculate the Jacobian vector product (the Hessian applied to an image) using SIRF
            back_sirf = ctx.objective_function.accumulate_Hessian_times_input(ctx.x_sirf, ctx.sirf_template_image, ctx.subset)

            # convert back to torch tensor via numpy
            back = torch.tensor(back_sirf.as_array(), device=ctx.device, dtype=ctx.dtype).unsqueeze(0).unsqueeze(0)

            return back, None, None, None
