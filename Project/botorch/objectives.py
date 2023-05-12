'''
Collection of test functions. Each has obj function call, as well as get_max(), get_domain(), get_points().
All are framed as maximization problems.
'''
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

import utils


Noise = float | Tensor | Callable[..., float | Tensor]
ObjectiveFunc = Callable[[Tensor, Noise], Tensor]

class Objective:
    name: str

    @staticmethod
    def objective(x: Tensor, noise: Noise = 0.) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args
            x: shape [batch_size, d], input
            noise:

        Returns: shape [batch_size]
        """
        raise NotImplemented

    @staticmethod
    def get_max() -> Tensor:
        """Returns maximum value of objective function."""
        raise NotImplemented

    @staticmethod
    def get_domain() -> tuple[Tensor, Tensor]:
        """Returns (low, high) domain of objective function.

        low, high have type FloatTensor.
        """
        raise NotImplemented

    @staticmethod
    def get_points() -> tuple[Tensor, Tensor]:
        """Returns (x, y) pairs."""
        raise NotImplemented

    @staticmethod
    def get_all_points() -> tuple[Tensor, Tensor]:
        """Returns (x, y) pairs."""
        raise NotImplemented


class Hartmann_6d(Objective):
    name = 'h6d'

    @staticmethod
    def objective(x: Tensor, noise: Noise = 0.) -> Tensor:
        alpha = Tensor([1.0, 1.2, 3.0, 3.2]).t()
        A = Tensor([[10, 3, 17, 3.5, 1.7, 8],
                          [0.05, 10, 17, 0.1, 8, 14],
                          [3, 3.5, 1.7, 10, 17, 8],
                          [17, 8, 0.05, 10, 0.1, 14]])
        P = 1e-4 * Tensor([[1312, 1696, 5569, 124, 8283, 5886],
                                 [2329, 4135, 8307, 3736, 1004, 9991],
                                 [2348, 1451, 3522, 2883, 3047, 6650],
                                 [4047, 8828, 8732, 5743, 1091, 381]])
        outer: Any = 0
        for i in range(4):
            inner: Any = 0
            for j in range(6):
                xj = x[:, j]
                Aij = A[i][j]
                Pij = P[i][j]
                inner += Aij * (xj - Pij)**2
            new = alpha[i] * torch.exp(-inner)
            outer += new
        # scaled
        # y = -(2.58 + outer) / 1.94
        # unscaled
        y = -outer
        if callable(noise):
            n = noise()
        else:
            n = noise
        if y.size(0) == 1:
            y = Tensor([y.item()])
        return (-y + n).float()

    @staticmethod
    def get_max() -> Tensor:
        maxx = Tensor([[.20169, .15001, .47687, .27533, .31165, .6573]])
        # 3.0425 scaled, 3.32237 unscaled
        maxy = Hartmann_6d.objective(maxx)
        return maxy

    @staticmethod
    def get_domain() -> tuple[Tensor, Tensor]:
        return torch.zeros(1,6).float(), torch.ones(1,6).float()

    @staticmethod
    def get_points(num_steps: int = 7) -> tuple[Tensor, Tensor]:
        x, y = utils.test_grid(Hartmann_6d.get_domain(), Hartmann_6d.objective, samp_per_dim=num_steps)
        return x, y

    @staticmethod
    def get_all_points() -> tuple[Tensor, Tensor]:
        return Hartmann_6d.get_points()


class Hartmann_3d(Objective):
    name = 'h3d'

    @staticmethod
    def objective(x: Tensor, noise: Noise = 0.) -> Tensor:
        alpha = Tensor([1.0, 1.2, 3.0, 3.2]).t()
        # [[3.0, 10, 30],[0.1, 10, 35],[3.0, 10, 30],[0.1, 10, 35]]
        A = Tensor([[3.0, 10, 30],[0.1, 10, 35],[3.0, 10, 30],[0.1, 10, 35]])
        P = 10**(-4) * Tensor([[3689, 1170, 2673],[4699, 4387, 7470],[1091, 8732, 5547],[381, 5743, 8828]])
        outer: Any = 0
        for i in range(4):
            inner: Any = 0
            for j in range(3):
                xj = x[:, j]
                Aij = A[i][j]
                Pij = P[i][j]
                inner += Aij * (xj - Pij)**2
            new = alpha[i] * torch.exp(-inner)
            outer += new
        y = -outer
        if callable(noise):
            n = noise()
        else:
            n = noise
        if y.size(0) == 1:
            y = Tensor([y.item()])
        return (-y + n).float()

    @staticmethod
    def get_max() -> Tensor:
        # this doesn't appear to be a true max, so just going to use 4 instead => can never hit 0 regret
        # maxx = Tensor([[.114614, .555649, .852547]])
        # # 3.86278, unscaled
        # maxy = Hartmann_3d.objective(maxx)
        # return maxy
        return Tensor([4.0]).float()

    @staticmethod
    def get_domain() -> tuple[Tensor, Tensor]:
        return torch.zeros(1,3).float(), torch.ones(1,3).float()

    @staticmethod
    def get_points(num_steps=50):
        x, y = utils.test_grid(Hartmann_3d.get_domain(), Hartmann_3d.objective, samp_per_dim=num_steps)
        return x, y

    @staticmethod
    def get_all_points() -> tuple[Tensor, Tensor]:
        return Hartmann_3d.get_points()


class SinePoly_1d(Objective):
    name = 'sinepoly'

    @staticmethod
    def objective(x: Tensor, noise: Noise = 0.) -> Tensor:
        y = x.float()**2 * torch.sin(2 * x.float())
        if callable(noise):
            n = noise()
        else:
            n = noise
        return (-Tensor([y.item()]) + n).float()

    @staticmethod
    def get_points(low=0, high=9, num_steps=100):
        x = torch.linspace(low, high, steps=num_steps)
        y = [SinePoly_1d.objective(inp) for inp in x]
        y = torch.cat(y, -1)
        return x, y

    @staticmethod
    def get_max(low=0, high=9):
        x, y = SinePoly_1d.get_points(low=low, high=high)
        return torch.max(y)

    @staticmethod
    def get_domain(low=0, high=9):
        return Tensor([low]).float(), Tensor([high]).float()

    @staticmethod
    def get_all_points() -> tuple[Tensor, Tensor]:
        return SinePoly_1d.get_points()


class Eggholder_2d(Objective):
    name = 'egg'

    @staticmethod
    def objective(x: Tensor, noise: Noise = 0.) -> Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        term1 = -(x2 + 47) * torch.sin(torch.sqrt(torch.abs(x2 + (x1 / 2) + 47)))
        term2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47))))
        y = term1 + term2
        if callable(noise):
            n = noise()
        else:
            n = noise
        return (-y + n).float()

    @staticmethod
    def get_max() -> Tensor:
        maxx = Tensor([[512, 404.2319]])
        # -959.6407
        maxy = Eggholder_2d.objective(maxx)
        return maxy

    @staticmethod
    def get_domain() -> tuple[Tensor, Tensor]:
        # -512, 512 both dim
        return Tensor([[-512, -512]]).float(), Tensor([[512, 512]]).float()

    @staticmethod
    def get_points(num_steps=200):
        x, y = utils.test_grid(Eggholder_2d.get_domain(), Eggholder_2d.objective, samp_per_dim=num_steps)
        return x, y

    @staticmethod
    def get_all_points() -> tuple[Tensor, Tensor]:
        return Eggholder_2d.get_points()

class Branin_2d(Objective):
    name = 'branin'

    @staticmethod
    def objective(x: Tensor, noise: Noise = 0.) -> Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        a, b, c, r, s, t = 1, 5.1/(4*torch.pi**2), 5/torch.pi, 6, 10, 1/(8*torch.pi)
        y = a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*torch.cos(x1) + s
        if callable(noise):
            n = noise()
        else:
            n = noise
        return (-y + n).float()

    @staticmethod
    def get_max() -> Tensor:
        # this doesn't appear to be true max... report 0?
        # # multiple global
        # maxx = Tensor([[-torch.pi, 12.275]])
        # # -0.3979
        # maxy = Branin_2d.objective(maxx)
        # return maxy
        return Tensor([0.0]).float()

    @staticmethod
    def get_domain() -> tuple[Tensor, Tensor]:
        # [-5,10] and [0,15]
        return Tensor([[-5, 0]]).float(), Tensor([[10, 15]]).float()

    @staticmethod
    def get_points(num_steps=200):
        x, y = utils.test_grid(Branin_2d.get_domain(), Branin_2d.objective, samp_per_dim=num_steps)
        return x, y

    @staticmethod
    def get_all_points() -> tuple[Tensor, Tensor]:
        return Branin_2d.get_points()

class DropWave_2d(Objective):
    name = 'dropwave'

    @staticmethod
    def objective(x: Tensor, noise: Noise = 0.) -> Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        a, b, c, r, s, t = 1, 5.1/(4*torch.pi**2), 5/torch.pi, 6, 10, 1/(8*torch.pi)
        y = a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*torch.cos(x1) + s
        if callable(noise):
            n = noise()
        else:
            n = noise
        return (-y + n).float()

    @staticmethod
    def get_max() -> Tensor:
        # multiple global
        maxx = Tensor([[-torch.pi, 12.275]])
        # -0.3979
        maxy = DropWave_2d.objective(maxx)
        return maxy

    @staticmethod
    def get_domain() -> tuple[Tensor, Tensor]:
        # [-5.12, 5.12] for both dim
        return Tensor([[-5.12, -5.12]]).float(), Tensor([[5.12, 5.12]]).float()

    @staticmethod
    def get_points(num_steps: int = 200) -> tuple[Tensor, Tensor]:
        x, y = utils.test_grid(DropWave_2d.get_domain(), DropWave_2d.objective, samp_per_dim=num_steps)
        return x, y

    @staticmethod
    def get_all_points() -> tuple[Tensor, Tensor]:
        return DropWave_2d.get_points()


##### real datasets

##########
# npx = '/scratch/ml/nano/50kRVs.pt'
# npy = '/scratch/ml/nano/50kFOMs_750nm.pt'
npx = '/scratch/ml/jbowden/nano/sf_sample/50kRVs.pt'
npy = '/scratch/ml/jbowden/nano/sf_sample/50kFOMs_750nm.pt'

class Nanophotonics(Objective):
    name = 'nano'

    # want to minimize FOM
    @staticmethod
    def objective(x: Tensor, noise: Noise = 0.) -> tuple[Tensor, Tensor]:
        X = torch.load(npx)
        y = torch.load(npy)#[:, 2:3]
        # need to return actual x queried, not one asked for in cont. opt case
        qx, qy = utils.query_discrete(X, -y, x)
        return qx.float(), qy.float()

    @staticmethod
    def get_max() -> Tensor:
        # has 3 fidelities for now, only want target (3rd)
        y = torch.load(npy)#[:, 2:3]
        return torch.max(-y).float()

    @staticmethod
    def get_domain() -> tuple[Tensor, Tensor]:
        X = torch.load(npx)
        lower, upper = utils.domain_discrete(X)
        return lower.float(), upper.float()

    @staticmethod
    def get_points() -> tuple[Tensor, Tensor]:
        X = torch.load(npx)
        y = torch.load(npy)#[:, 2:3]
        return X.float(), -y.float()


#############################
tdx = '/scratch/ml/FoldX/cleaned_coronavirus_dump_m396_20200901_1400_x_foldx_trunc.pt'
tdy = '/scratch/ml/FoldX/cleaned_coronavirus_dump_m396_20200901_1400_y_foldx.pt'

class FoldX(Objective):
    name = 'foldx'

    @staticmethod
    def objective(x: Tensor, noise: Noise = 0.) -> tuple[Tensor, Tensor]:
        X = torch.load(tdx)
        y = torch.load(tdy)
        # need to return actual x queried, not one asked for
        qx, qy = utils.query_discrete(X, -y, x)
        return qx.float(), qy.float()

    @staticmethod
    def get_max() -> Tensor:
        y = torch.load(tdy)
        return torch.max(-y).float()

    @staticmethod
    def get_domain() -> tuple[Tensor, Tensor]:
        X = torch.load(tdx)
        lower, upper = utils.domain_discrete(X)
        return lower.float(), upper.float()

    @staticmethod
    def get_points() -> tuple[Tensor, Tensor]:
        X = torch.load(tdx)
        y = torch.load(tdy)
        return X.float(), -y.float()

    @staticmethod
    def get_all_points() -> tuple[Tensor, Tensor]:
        return FoldX.get_points()

# bwx = '/scratch/ml/GB1/50kGB1_x_filt.pt'
# bwy = '/scratch/ml/GB1/50kGB1_y_filt.pt'

class GB1(Objective):
    name = 'gb1'

    def __init__(self, encoding):
        if encoding == 'onehot':
            self.bwx = '/home/jyang4/repos/data/GB1_onehot_x.pt'
            self.bwy = '/home/jyang4/repos/data/GB1_onehot_y.pt'
        elif encoding == 'ESM1b':
            self.bwx = '/home/jyang4/repos/data/GB1_ESM1b_x.pt'
            self.bwy = '/home/jyang4/repos/data/GB1_ESM1b_y.pt'
        elif encoding == 'TAPE':
            self.bwx = '/home/jyang4/repos/data/GB1_TAPE_x.pt'
            self.bwy = '/home/jyang4/repos/data/GB1_TAPE_y.pt'

    def objective(self, x: Tensor, noise: Noise = 0.) -> tuple[Tensor, Tensor]:

        X = torch.load(self.bwx)
        y = torch.load(self.bwy)
        # need to return actual x queried, not one asked for
        qx, qy = utils.query_discrete(X, y, x)
        return qx.float(), qy.float()

    def get_max(self) -> Tensor:
        y = torch.load(self.bwy)
        return torch.max(y).float()

    def get_domain(self) -> tuple[Tensor, Tensor]:
        X = torch.load(self.bwx)
        lower, upper = utils.domain_discrete(X)
        return lower.float(), upper.float()

    def get_points(self) -> tuple[Tensor, Tensor]:
        X = torch.load(self.bwx)
        y = torch.load(self.bwy)
        return X.float(), y.float()

    @staticmethod
    def get_all_points() -> tuple[Tensor, Tensor]:
        return GB1.get_points()


ALL_OBJS = [
    Hartmann_6d, Hartmann_3d, SinePoly_1d, Eggholder_2d, Branin_2d,
    DropWave_2d, Nanophotonics, FoldX, GB1]

NAME_TO_OBJ = {
    obj.name: obj for obj in ALL_OBJS
}
