"""skeleton for stir-pytorch layer that allows to build a listmode EM update layer"""

# Questions 
# =========  
#        1) Can we use SIRF or do we need STIR? -> Depends on whether "Hessian applied to input of LM-logL"
#           is available in SIRF 
#        2) forward projection of current image x is needed in forward an backward pass
#           -> can we cache this in a pytorch context (ctx) object to avoid recomputing it in the backward pass?

# INPUT: 1) pytorch tensor containing the current image estimate (x)
#        2) SIRF or STIR Poisson logL cost function object that can compute the gradient w.r.t. to
#           the image estimate (x) and the Hessian at x applied to another image x'


# SKELETON
# ========

# (1) setup auto-grad comp. layer that returns grad LM-logL as output (gradient descent step with step size 1)
#     -> needs "Hessian" of LM-logL applied to x' for back propagation

# (2) setup EM_update module that uses x + x/s * grad logL in forward pass
#     -> let autograd compute the gradients


# PSEUDOCODE FOR (1)
# ==================

# forward pass
# INPUT: x (torch tensor), LM-logL (sirf / stir listmode Poisson logL)
# 1.1: convert x from torch (GPU) tensor to sirf / stir (CPU) image -> probably via numpy
# 1.2: compute gradient of SIRF/STIR LM-logL
# 1.3: convert output from sirf / stir image back to (GPU) torch tensor 

# backward pass
# INPUT: grad_output (torch tensor), LM-logL (sirf / stir listmode Poisson logL)
# 1.4: convert grad_output from (GPU) torch tensor to (CPU) sirf / stir image (x') -> probably via numpy
# 1.5: apply Hessianof SIRF/STIR LM-logL at x to x'
# 1.6: convert output from sirf / stir image back to (GPU) torch tensor -> probably via numpy


# %%
import torch
from lm_data_fid import test_lmlogl

class LMPoissonLogLGradLayer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, lm_objective_list, 
    ) -> torch.Tensor:

        ctx.set_materialize_grads(False)
        ctx.x = x.detach()
        ctx.lm_objective_list = lm_objective_list

        y = torch.zeros_like(x)

        for i in range(x.shape[0]):
            y[i, 0, ...] = lm_objective_list[i].gradient(ctx.x[i, 0, ...])

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        if grad_output is None:
            return None, None
        else:
            back = torch.zeros_like(grad_output)

            for i in range(grad_output.shape[0]):
                back[i, 0, ...] = ctx.lm_objective_list[i].hessian_applied(ctx.x[i, 0, ...], grad_output[i, 0, ...].detach())

            return back, None

# %%
class LMEMNet(torch.nn.Module):  
    def __init__(self, lm_obj_list, num_blocks: int = 20) -> None:
        super().__init__()
        self._data_fid_gradient_layer = LMPoissonLogLGradLayer.apply
        self._lm_obj_list = lm_obj_list
        self._sens_imgs = torch.stack([l.op.adjoint_ones.unsqueeze(0) for l in lm_obj_list])
        self._num_blocks = num_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for _ in range(self._num_blocks):
            x =  x + (x / self._sens_imgs) * self._data_fid_gradient_layer(x, lm_obj_list)

        return x


# %%
torch.manual_seed(0)
if torch.cuda.is_available():
    #dev = "cuda:0"
    dev = "cpu"
else:
    dev = "cpu"


# %%

# setup a list of dummy LM objective function corresponding to a mini-batch of LM acquisitions
lm_obj1 = test_lmlogl(dev, nx = 4, ny = 4)
lm_obj2 = test_lmlogl(dev, nx = 4, ny = 4)
lm_obj_list = [lm_obj1, lm_obj2]
batch_size = len(lm_obj_list)

# setup a test input mini-batch of images that we use to test our network
x_t = torch.rand(
    (batch_size, 1) + lm_obj_list[0].op.in_shape,
    device=dev,
    dtype=torch.float64,
    requires_grad=True,
)

# setup the LM Network using 20 blocks (20 LM EM updates)
lmem_net = LMEMNet(lm_obj_list, num_blocks=20)

# feed out test image through the network
x_fwd = lmem_net(x_t)
# calculate the analytic ML solution (possible since we have invertible 2x2 forward operators) 
x_ml = torch.stack([(torch.linalg.inv(l.op.A) @ l.op.data.double()).unsqueeze(0) for l in lm_obj_list], dim=0)

# test the gradient back propagation through the network
test_lmemnet = torch.autograd.gradcheck(lmem_net, (x_t,))