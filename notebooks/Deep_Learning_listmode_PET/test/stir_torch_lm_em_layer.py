# %% [markdown]
# Skeleton for STIR-based listmode Poisson logL gradient data fidelity layer
# ==========================================================================
#
# Hello 3
# $$\lambda^n = 1$$
#

# %%
import torch
from lm_data_fid import test_lmlogl


class LMPoissonLogLGradLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lm_objective_list) -> torch.Tensor:

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
                back[i, 0, ...] = ctx.lm_objective_list[i].hessian_applied(
                    ctx.x[i, 0, ...], grad_output[i, 0, ...].detach()
                )

            return back, None


# %%
class LMEMNet(torch.nn.Module):
    def __init__(self, lm_obj_list, num_blocks: int = 20) -> None:
        super().__init__()
        self._data_fid_gradient_layer = LMPoissonLogLGradLayer.apply
        self._lm_obj_list = lm_obj_list
        self._sens_imgs = torch.stack(
            [l.op.adjoint_ones.unsqueeze(0) for l in lm_obj_list]
        )
        self._num_blocks = num_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for _ in range(self._num_blocks):
            x = x + (x / self._sens_imgs) * self._data_fid_gradient_layer(
                x, lm_obj_list
            )

        return x


# %%
torch.manual_seed(0)
if torch.cuda.is_available():
    # dev = "cuda:0"
    dev = "cpu"
else:
    dev = "cpu"


# %%

# setup a list of dummy LM objective function corresponding to a mini-batch of LM acquisitions
lm_obj1 = test_lmlogl(dev, nx=4, ny=4)
lm_obj2 = test_lmlogl(dev, nx=4, ny=4)
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
x_ml = torch.stack(
    [(torch.linalg.inv(l.op.A) @ l.op.data.double()).unsqueeze(0) for l in lm_obj_list],
    dim=0,
)

# test the gradient back propagation through the network
test_lmemnet = torch.autograd.gradcheck(lmem_net, (x_t,))
