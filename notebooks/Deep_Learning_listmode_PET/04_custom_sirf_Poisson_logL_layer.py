# %% [markdown]
# Learning objectives
# ===================
#
# 1. Basics of pytorch's automatic gradient calculation (autograd)
# 2. How to implement the "backward" pass for a layer that perform a
#    gradient ascent step according the the PET objective function (data fidelity)
# 3. How to test whether the backward pass is implemented correctly (torch.autograd.gradcheck). 

# %%
import sirf.STIR
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sirf.Utilities import examples_data_path

data_path: Path = Path(examples_data_path("PET")) / "mMR"
list_file: str = str(data_path / "list.l.hdr")
nxny: tuple = (7, 7)
num_subsets: int = 21

# engine's messages go to files, except error messages, which go to stdout
_ = sirf.STIR.MessageRedirector("info.txt", "warn.txt")

# %%
sirf.STIR.AcquisitionData.set_storage_scheme("memory")
listmode_data = sirf.STIR.ListmodeData(list_file)
acq_data_template = listmode_data.acquisition_data_template()
print(acq_data_template.get_info())

# %%
# select acquisition model that implements the geometric
# forward projection by a ray tracing matrix multiplication
acq_model = sirf.STIR.AcquisitionModelUsingRayTracingMatrix()
acq_model.set_num_tangential_LORs(1)

# %%
initial_image = acq_data_template.create_uniform_image(value=1, xy=nxny)

# %%
lm_obj_fun = (
    sirf.STIR.PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin()
)
lm_obj_fun.set_acquisition_model(acq_model)
lm_obj_fun.set_acquisition_data(listmode_data)
lm_obj_fun.set_num_subsets(num_subsets)
print('setting up listmode objective function ...')
lm_obj_fun.set_up(initial_image)

# %%
#g = lm_obj_fun.gradient(initial_image, subset=0)
#hess_out_img_lm = lm_obj_fun.accumulate_Hessian_times_input(initial_image, 2*initial_image, subset=0)



# %%

class PoissonlogLGradientLayer(torch.nn.Module):  
    def __init__(self, obj_fun, template_image: sirf.STIR.ImageData, subset: int = 0) -> None:
        super().__init__()
        self._obj_fun = obj_fun
        self._x_sirf = template_image.clone()
        self._subset = subset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # convert torch tensor to numpy array and strip the batch and channel dimensions
        x_np: np.ndarray = x.numpy(force = True)[0,0,...]
        self._x_sirf.fill(x_np)
        g: sirf.STIR.ImageData = self._obj_fun.gradient(self._x_sirf, subset=self._subset)

        # convert to back to torch tensor and add batch and channel dimensions
        y = torch.tensor(g.as_array(), device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)

        return y

# %%
class UnrolledPoissonlogLGradientNet(torch.nn.Module):  
    def __init__(self, obj_fun, template_image: sirf.STIR.ImageData, subsets: list[int], device: str, step_sizes = None) -> None:
        super().__init__()
        self._poisson_logL_grad_layers = torch.nn.ModuleList(
            [
                PoissonlogLGradientLayer(obj_fun, template_image, subset=sub)
                for sub in subsets
            ]
        )

        if step_sizes is None:
            self._step_sizes = [1.0] * len(subsets)
        else:
            self._step_sizes = step_sizes

        self._conv_net = torch.nn.Conv3d(1, 1, (3,3,3), dtype=torch.float32, device = device, padding='same')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        for i, layer in enumerate(self._poisson_logL_grad_layers):
            y += self._step_sizes[i] * layer(y) #+ self._conv_net(y)
        return y


# %%
torch.manual_seed(0)
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

batch_size = 1

# setup a test input mini-batch of images that we use to test our network
# the dimension are [batch, channel, spatial dimensions]
x_t = torch.rand(
    (batch_size, 1) + initial_image.shape,
    device=dev,
    dtype=torch.float32,
    requires_grad=False,
)

# %%
my_net = UnrolledPoissonlogLGradientNet(lm_obj_fun, initial_image, subsets=[0,1], device = dev)
prediction = my_net(x_t)


# %%
target = torch.ones_like(x_t, requires_grad=False)

loss_fct = torch.nn.MSELoss()
loss = loss_fct(prediction, target)
loss.backward()