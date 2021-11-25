import abc
import torch
import torch.nn as nn
from torch.autograd import grad


def vjp(z, x, v_like_z=None):
    """Computes the vector-Jacobian product vjp = v^T dot dz/dx."""
    if v_like_z is None:
        v_like_z = torch.ones_like(z).float().requires_grad_(True)
    eval_vjp = grad(z, x, v_like_z, create_graph=True, allow_unused=True)
    return eval_vjp[0]


def jvp(z, x, v_like_x=None):
    """Computes the Jacobian-vector product jvp = dz/dx dot v.

    Trick: jvp(z, x, v) = vjp(vjp(z, x, u)^T, u, v))^T for a dummy `u`

    See `https://j-towns.github.io/2017/06/12/A-new-trick.html`
    """
    if v_like_x is None:
        v_like_x = torch.ones_like(x).float().requires_grad_(True)
    v_like_z = torch.ones_like(z).float().requires_grad_(True)
    vjpt = vjp(z, x, v_like_z)  # No explicit transpose due to vector output
    jvpt = grad(vjpt, v_like_z, v_like_x, create_graph=True, allow_unused=True)
    return jvpt[0]


class PhiLayers(nn.Module):
    """Controls the network parameters with respect to Phi."""

    def __init__(self,
                 dim: int,
                 num_layers: int,
                 activation,
                 double_mlp: bool = False,
                 triple_mlp: bool = False,
                 bias_on: bool = True,
                 mult: int = 1,
                 ):
        super(PhiLayers, self).__init__()
        self.double_mlp = double_mlp
        self.triple_mlp = triple_mlp if double_mlp else False
        self.activation = activation
        self.num_evals = 0

        if double_mlp:
            self.single_layers = nn.ModuleList(
                [nn.Linear(dim, (mult * dim), bias=bias_on)
                 for _ in range(num_layers)]
            )
            if triple_mlp:
                self.triple_layers = nn.ModuleList(
                    [nn.Linear((mult * dim), dim, bias=bias_on)
                     for _ in range(num_layers)]
                )
                self.double_layers = nn.ModuleList(
                    [nn.Linear((mult * dim), (mult * dim), bias=bias_on)
                     for _ in range(num_layers)]
                )
            else:
                self.double_layers = nn.ModuleList(
                    [nn.Linear((mult * dim), dim, bias=bias_on)
                     for _ in range(num_layers)]
                )
        else:
            self.single_layers = nn.ModuleList(
                [nn.Linear(dim, dim, bias=bias_on)
                 for _ in range(num_layers)]
            )

    def forward(self, ell, x):
        """Note that the ell-1 index runs from 0,...,ell_max-1."""
        self.num_evals += 1
        out = self.single_layers[ell-1](self.activation(x))
        if self.double_mlp:
            out = self.double_layers[ell-1](self.activation(out))
        if self.triple_mlp:
            out = self.triple_layers[ell-1](self.activation(out))
        return out


class InImNetDisc(nn.Module, metaclass=abc.ABCMeta):
    """A `Discrete Step` Invariant Imbedding Network."""

    def __init__(self,
                 input_dim: int,
                 ell_max: int,
                 activation,
                 double_mlp: bool = False,
                 triple_mlp: bool = False,
                 bias_on: bool = True,
                 dim_multiplier: int = 1,
                 cost_fn=nn.MSELoss(reduction='none'),
                 ):
        super(InImNetDisc, self).__init__()
        self.dim = input_dim
        self.ell_max = ell_max
        self.bias_on = bias_on
        self.cost_fn = cost_fn
        self.phi = PhiLayers(dim=input_dim,
                             num_layers=ell_max,
                             activation=activation,
                             double_mlp=double_mlp,
                             triple_mlp=triple_mlp,
                             bias_on=bias_on,
                             mult=dim_multiplier,
                             )
        self.vjp_evals = 0
        self.jvp_evals = 0

    def forward(self, x, inim_on: bool = False):
        """Returns an (ell_max+1) list containing z(t; ell, x) in two cases.

            (1) If `inim_on = True` then z_ell[ell] = z(0; ell, x).
            (2) If `inim_on = False` then z_tee[-t] = z(t; ell_max, x).
        """

        if inim_on:
            z_ell = [x]
            for ell in range(self.ell_max):
                # Performs Euler step of size 1 for the ell-gradient of z(0).
                z_ell.append(z_ell[-1] + jvp(z_ell[-1], x, self.phi(ell+1, x)))
                self.jvp_evals += 1
            return z_ell
        else:
            z_tee = [x]
            for t in range(-self.ell_max, 0):
                # Performs a Euler step of size 1 for a resnet implementation.
                z_tee.append(z_tee[-1] + self.phi(-t, z_tee[-1]))
            z_tee.reverse()  # list indexed from -t = 0, ... , ell_max
            return z_tee

    @abc.abstractmethod
    def output(self, z0):
        """The output layer: to reduce feature maps to target dimension."""
        return z0

    def _inim_lambda(self, x, y):
        """Integrate Lambda(ell, x) wrt ell via a step size-1 Euler method."""
        init_loss = self.cost_fn(self.output(x), y)
        lam = [vjp(init_loss, x)]
        for ell in range(self.ell_max):
            # Performs a Euler step of size 1 on the ell-gradient of lambda.
            lam.append(lam[ell] + jvp(lam[ell], x, self.phi(ell+1, x))
                                + vjp(self.phi(ell+1, x), x, lam[ell])
                       )
            self.vjp_evals += 1
            self.jvp_evals += 1
        return lam

    def _resnet_lambda(self, z, y):
        """Backpropagation of lambda(t; ell_max, x) wrt t."""
        init_loss = self.cost_fn(self.output(z[0]), y)
        lam = [vjp(init_loss, z[0])]
        for t in range(0, -self.ell_max, -1):
            # Performs a Euler step of size 1 on the ell-gradient of lambda.
            next_z = z[-t + 1]
            lam.append(lam[-1] + vjp(self.phi(-t+1, next_z), next_z, lam[-1]))
            self.vjp_evals += 1
        return lam  # list indexed from -t = 0, ... , ell_max (same as z)

    def loss_grad_wrt_params(self, x_or_z, y, inim_on):
        """Relate lambda(-ell; ell, x) to loss at the ell-th param layer."""
        loss_grad_list = list()

        if inim_on:
            x = x_or_z
            lam = self._inim_lambda(x, y)
            for ell0, param in enumerate(self.parameters()):
                if self.bias_on:
                    ell0 = ell0//2  # Every other parameter, skipping biases
                ell0 = ell0 % self.ell_max  # Shift for double/triple layers
                ell = ell0 + 1  # Correct index ell0 = 0, ... , ell_max-1
                loss_grad_list.append(
                    vjp(z=self.phi(ell, x), x=param, v_like_z=(
                        lam[ell-1]
                        + jvp(z=lam[ell-1], x=x, v_like_x=self.phi(ell, x))),
                        )
                )
                self.vjp_evals += 1
                self.jvp_evals += 1
        else:
            z = x_or_z
            lam = self._resnet_lambda(z, y)
            for ell0, param in enumerate(self.parameters()):
                if self.bias_on:
                    ell0 = ell0 // 2  # Every other parameter, skipping biases
                ell0 = ell0 % self.ell_max  # Shift for double/triple layers
                t = -(ell0 + 1)  # Correct index ell0 = 0, ... , ell_max-1
                loss_grad_list.append(
                    vjp(z=self.phi(-t, z[-t]), x=param, v_like_z=lam[-t-1])
                )
                self.vjp_evals += 1

        return loss_grad_list

    @property  # Called upon with each call to self.nfe
    def func_evals(self):
        # Number of evaluations of (phi, vjp, jvp)
        return self.phi.num_evals, self.vjp_evals, self.jvp_evals

    @func_evals.setter  # Called upon with self.func_evals = setter_value
    def func_evals(self, setter_value):
        self.phi.num_evals = setter_value
        self.vjp_evals = setter_value
        self.jvp_evals = setter_value
