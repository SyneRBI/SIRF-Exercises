import torch


class LMLinearOperator:
    def __init__(self, A: torch.tensor) -> None:
        self._A = A
        self._lm_data = None
        self._adjoint_ones = None

    @property
    def in_shape(self) -> tuple[int]:
        return (self._A.shape[1],)

    @property
    def out_shape(self) -> tuple[int]:
        return (self._A.shape[0],)

    @property
    def A(self) -> torch.tensor:
        return self._A

    @property
    def lm_data(self) -> torch.tensor:
        return self._lm_data

    @lm_data.setter
    def lm_data(self, value: torch.tensor) -> None:
        self._lm_data = value

    @property
    def adjoint_ones(self) -> torch.tensor:
        if self._adjoint_ones is None:
            self._adjoint_ones = self._A.T @ torch.ones(
                self._A.shape[0], device=self._A.device, dtype=torch.float64
            )
        return self._adjoint_ones

    @property
    def data(self) -> torch.tensor:
        data = torch.zeros(self._A.shape[0], device=self._A.device, dtype=torch.int)
        for i in range(self._A.shape[0]):
            data[i] = (self._lm_data == i).sum().item()

        return data

    def __call__(self, x: torch.tensor) -> torch.tensor:
        return self.fwd(x)

    def fwd(self, x: torch.tensor) -> torch.tensor:
        return self._A @ x

    def adjoint(self, y: torch.tensor) -> torch.tensor:
        return self._A.T @ y.double()

    def fwd_lm(self, x: torch.tensor) -> torch.tensor:
        if self._lm_data is None:
            raise ValueError("must set lm data first")
        return self._A[self._lm_data, :] @ x

    def adjoint_lm(self, lst: torch.tensor) -> torch.tensor:
        return self._A[self._lm_data, :].T @ lst.double()


class PoissonLogL:
    def __init__(self, data: torch.tensor, op: LMLinearOperator) -> None:
        self._data = data
        self._op = op

    def __call__(self, x: torch.tensor) -> float:
        exp = self._op.fwd(x)
        return float((self._data * torch.log(exp) - exp).sum())

    def gradient(self, x: torch.tensor) -> torch.tensor:
        exp = self._op.fwd(x)
        return self._op.adjoint((self._data / exp) - 1)

    def hessian_applied(self, x: torch.tensor, x_prime: torch.tensor) -> torch.tensor:
        exp = self._op.fwd(x)
        exp_prime = self._op.fwd(x_prime)
        return -self._op.adjoint(self._data * exp_prime / (exp ** 2))


class LMPoissonLogL:
    def __init__(self, op: LMLinearOperator) -> None:
        self._op = op

    @property
    def op(self) -> LMLinearOperator:
        return self._op

    def gradient(self, x: torch.tensor) -> torch.tensor:
        exp = self._op.fwd_lm(x)
        return self._op.adjoint_lm(1 / exp) - self._op.adjoint_ones
        pass

    def hessian_applied(self, x: torch.tensor, x_prime: torch.tensor) -> torch.tensor:
        exp = self._op.fwd_lm(x)
        exp_prime = self._op.fwd_lm(x_prime)
        return -self._op.adjoint_lm(exp_prime / (exp ** 2))


def test_lmlogl(dev: str, nx: int = 2, ny: int = 2):

    x = torch.rand(nx, device=dev, dtype=torch.float64)
    op = LMLinearOperator(
        6 * torch.rand(ny, nx, device=dev, dtype=torch.float64)
        + 24 * torch.eye(ny, nx, device=dev, dtype=torch.float64)
    )

    y_noisefree = op(x)
    y = torch.poisson(y_noisefree).int()

    lm_data = torch.repeat_interleave(torch.arange(ny, device=dev), y)

    # shuffle the LM data using a random permuation
    shuffled_inds = torch.randperm(lm_data.shape[0])
    lm_data = lm_data[shuffled_inds]

    op.lm_data = lm_data
    lmdata_fid = LMPoissonLogL(op)

    return lmdata_fid
