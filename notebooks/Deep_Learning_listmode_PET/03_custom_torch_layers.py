# %% [markdown]
# Creating custom layers in pytorch
# =================================
#
# In this notebook, we will learn how to create custom layers in pytorch that use functions outside the pytorch framework.
# We will create a custom layer that multiplies the input tensor with a square matrix.
# For demonostration purposes, we will create a simple layer that multiplies a 1D torch input vector with a square matrix,
# where the matrix multiplication is done using numpy functions.
#
# Learning objectives of this notebook
# ------------------------------------
#
# 1. Learn how to create custom layers in pytorch that are compatible with the autograd framework.
# 2. Understand the importance of implementing the backward pass of the custom layer correctly.
# 3. Learn how to test the gradient backpropagation through the custom layer using the `torch.autograd.gradcheck` function.

# %%
# import modules
import torch
import numpy as np

# seed all torch random generators
torch.manual_seed(0)

# choose the torch device
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

# length of the input vector
n = 7

# define our square matrix
A: np.ndarray = np.arange(n ** 2).reshape(n, n).astype(np.float64) / (n ** 2)
# define the 1D pytorch tensor: not that the shape is (1,1,n) including the batch and channel dimensions
x_t = torch.tensor(np.arange(n).reshape(1, 1, n).astype(np.float64), device=dev) / n

# %% [markdown]
# Approach 1: The naive approach
# ------------------------------
#
# We will first try a naive approach where we create a custom layer by subclassing torch.nn.Module
# and implementing the forward pass by conversion between numpy and torch tensors.

# %%
class SquareMatrixMultiplicationLayer(torch.nn.Module):
    def __init__(self, mat: np.ndarray) -> None:
        super().__init__()
        self._mat: np.ndarray = mat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # convert the input tensor to numpy
        x_np = x.detach().cpu().numpy()
        # nympy matrix multiplication
        y_np = self._mat @ x_np[0, 0, ...]
        # convert back to torch tensor
        y = torch.tensor(y_np, device=x.device).unsqueeze(0).unsqueeze(0)

        return y


# %% [markdown]
# We setup a simple feedforward network interlacing the 3 convolutional 3 square matrix multiplication layers.

# %%
class Net1(torch.nn.Module):
    def __init__(self, mat, cnn) -> None:
        super().__init__()
        self._matrix_layer = SquareMatrixMultiplicationLayer(mat)
        self._cnn = cnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self._cnn(x)
        x2 = self._matrix_layer(x1)
        x3 = self._cnn(x2)
        x4 = self._matrix_layer(x3)
        x5 = self._cnn(x4)
        x6 = self._matrix_layer(x5)

        return x6


# %%
# setup a simple CNN consisting of 2 convolutional layers and 1 ReLU activation
cnn1 = torch.nn.Sequential(
    torch.nn.Conv1d(1, 3, (3,), padding="same", bias=False, dtype=torch.float64),
    torch.nn.ReLU(),
    torch.nn.Conv1d(3, 1, (3,), padding="same", bias=False, dtype=torch.float64),
).to(dev)

# setup the network
net1 = Net1(A, cnn1)

# %%
# forward pass of our input vector through the network
pred1 = net1(x_t)
print(f"pred1: {pred1}\n")


# %% [markdown]
# We see that the forward pass works as expected. Now we will setup a dummy loss and try backpropagate the gradients
# using the naive approach for our custom matrix multiplication layer.
# Baclpropagation of the gradients is the central step in training neural networks. It involves calculating the gradients of
# the loss function with respect to the weights of the network.

# %%
# setup a dummy target (label / high quality reference image) tensor
target = 2 * x_t
# define an MSE loss
loss_fct = torch.nn.MSELoss()
# calculate the loss between the prediction and the target
loss1 = loss_fct(pred1, target)
print(f"loss1: {loss1.item()}\n")

# %% [markdown]
# Calculation of the loss still runs fine. Now let's try to backpropagate the gradients.

# %%
try:
    loss1.backward()
except RuntimeError:
    print("Error in gradient backpropagation using naive approach\n")

# %% [markdown]
# Exercise 3.1
# ------------
# We see that the backpropagation of the gradients fails with the naive approach.
# Why is that?


# %% [markdown]
# Approach 2: Subclassing torch.autograd.Function
# -----------------------------------------------
#
# The correct way to create custom layers in pytorch is to subclass torch.autograd.Function
# which involves implementing the forward and backward pass of the layer.
# In the backward pass we have to implement the Jacobian transpose vector product of the layer.
# For details, see [here](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#optional-reading-vector-calculus-using-autograd)
# and [here](https://pytorch.org/docs/stable/notes/extending.func.html).


# %%
# define the custom layer by subclassing torch.autograd.Function and implementing the forward and backward pass
class NPSquareMatrixMultiplicationLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, mat: np.ndarray) -> torch.Tensor:

        # we use the context object ctx to store the matrix and other variables that we need in the backward pass
        ctx.mat = mat
        ctx.device = x.device
        ctx.shape = x.shape
        ctx.dtype = x.dtype

        # convert to numpy
        x_np = x.cpu().numpy()
        # numpy matrix multiplication
        y_np = mat @ x_np[0, 0, ...]
        # convert back to torch tensor
        y = torch.tensor(y_np, device=ctx.device).unsqueeze(0).unsqueeze(0)

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        if grad_output is None:
            return None, None
        else:
            # convert to numpy
            grad_output_np = grad_output.cpu().numpy()
            # calculate the Jacobian transpose vector product in numpy and convert back to torch tensor
            back = (
                torch.tensor(
                    ctx.mat.T @ grad_output_np[0, 0, ...],
                    device=ctx.device,
                    dtype=ctx.dtype,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            return back, None


# %%
# define a new network incl. the custom matrix multiplication layer using the "correct" approach
# To use our custom layer in the network, we have to use the apply method of the custom layer class.
class Net2(torch.nn.Module):
    def __init__(self, mat, cnn) -> None:
        super().__init__()
        self._matrix_layer = NPSquareMatrixMultiplicationLayer.apply
        self._mat = mat
        self._cnn = cnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self._cnn(x)
        x2 = self._matrix_layer(x1, self._mat)
        x3 = self._cnn(x2)
        x4 = self._matrix_layer(x3, self._mat)
        x5 = self._cnn(x4)
        x6 = self._matrix_layer(x5, self._mat)

        return x6

# %%
# setup the same CNN as above
cnn2 = torch.nn.Sequential(
    torch.nn.Conv1d(1, 3, (3,), padding="same", bias=False, dtype=torch.float64),
    torch.nn.ReLU(),
    torch.nn.Conv1d(3, 1, (3,), padding="same", bias=False, dtype=torch.float64),
).to(dev)
cnn2.load_state_dict(cnn1.state_dict())

# setup the network - only difference is the custom layer
net2 = Net2(A, cnn2)

# predict again
pred2 = net2(x_t)
print(f"pred2: {pred2}\n")

loss2 = loss_fct(pred2, target)
print(f"loss2: {loss2.item()}\n")

# %% [markdown]
# Note that the prediction still works and gives the same result as before. Also the loss calculation yield the same results as before.

# %%
loss2.backward()

# print backpropagated gradients that of all parameters of CNN layers of our network
print("backpropagated gradients using correct approach")
print([p.grad for p in net2._cnn.parameters()])

# %% [markdown]
# In contrast to the naive approach, the backpropagation of the gradients works fine now, meaning that this network is ready for training.

# %% [markdown]
# Testing gradient backpropagation through the layer
# --------------------------------------------------
#
# When defining new custom layers, it is crucial to test whether the backward pass is implemented correctly.
# Otherwise the gradient backpropagation though the layer will be incorrect, and optimizing the model parameters will not work.
# To test the gradient backpropagation, we can use the `torch.autograd.gradcheck` function.

# %%
# setup a test input tensor - requires grad must be True!
t_t = torch.rand(x_t.shape, device=dev, dtype=torch.float64, requires_grad=True)

# test the gradient backpropagation through the custom numpy matrix multiplication layer
matrix_layer = NPSquareMatrixMultiplicationLayer.apply
gradcheck = torch.autograd.gradcheck(matrix_layer, (t_t, A), fast_mode=True)

print(f"gradient check of NPSquareMatrixMultiplicationLayer: {gradcheck}")

# %% [markdown]
# Exercise 3.2
# ------------
# Temporarily change the backward pass of the custom layer such that is is not correct anymore
# (e.g. by multiplying the output with 0.95) and rerun the gradient check. What do you observe?

# %% [markdown]
# WARNING
# -------
#
# Depending on the implementation of your custom layer, pytorch might not raise an error even if
# the backward pass is not implemented. This can lead to incorrect gradient backpropagation.
# Make sure to always test the gradient backpropagation through your custom layer using the `torch.autograd.gradcheck` function.
