import abc
import torch
import torch.nn as nn
from torch.autograd import grad
from torchdiffeq import odeint


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


class PhiCnstParams(nn.Module):
    """Controls the constant network parameters with respect to Phi."""

    def __init__(self,
                 dim: int,
                 activation,
                 double_mlp: bool=False,
                 triple_mlp: bool=False,
                 bias_on: bool=True,
                 mult: int=1,
                 ):
        super(PhiCnstParams, self).__init__()
        self.double_mlp = double_mlp
        self.triple_mlp = triple_mlp if double_mlp else False
        self.activation = activation
        self.num_evals = 0

        if double_mlp:
            self.single_layers = nn.Linear(dim, (mult * dim), bias=bias_on)

            if triple_mlp:
                self.triple_layers = nn.Linear(
                    (mult * dim), dim, bias=bias_on
                )
                self.double_layers = nn.Linear(
                    (mult * dim), (mult * dim), bias=bias_on
                )

            else:
                self.double_layers = nn.Linear((mult * dim), dim, bias=bias_on)
        else:
            self.single_layers = nn.Linear(dim, dim, bias=bias_on)

    def forward(self, p, x):
        """This example function is p-independant."""
        self.num_evals += 1
        out = self.single_layers(self.activation(x))
        if self.double_mlp:
            out = self.double_layers(self.activation(out))
        if self.triple_mlp:
            out = self.triple_layers(self.activation(out))
        return out


class InImNetCont(nn.Module, metaclass=abc.ABCMeta):
    """A `Discrete Step` Invariant Imbedding Network.

    Args:
        integration_method: input to odeint; choose from
            Adaptive: 'dopri8', 'dopri5', 'bosh3', 'fehlberg2',
            'adaptive_heun',
            Fixed point: 'euler', 'midpoint', 'rk4', 'explicit_adams',
            'implicit_adams',
            Backwards compatibility: 'fixed_adams', 'scipy_solver'
    """

    def __init__(self,
                 input_dim: int,
                 batch_dim: int,
                 cost_fn=nn.MSELoss(reduction='none'),
                 activation=nn.ReLU(inplace=True),
                 double_mlp: bool = False,
                 triple_mlp: bool = False,
                 dim_multiplier: int = 1,
                 bias_on: bool = True,
                 integration_method: str = 'euler',
                 ):
        super(InImNetCont, self).__init__()
        self.input_dim = input_dim
        self.batch_dim = batch_dim
        self.cost_fn = cost_fn
        self.integration_method = integration_method
        self.phi = PhiCnstParams(dim=input_dim,
                                 activation=activation,
                                 double_mlp=double_mlp,
                                 triple_mlp=triple_mlp,
                                 bias_on=bias_on,
                                 mult=dim_multiplier,
                                 )
        self.vjp_evals = 0
        self.jvp_evals = 0

    def lam_aug_dynamics(self, p, x, lam_aug, t_grad_on):
        """Implements the p-derivative of the adjoint Lambda(p, x).

        Args:
            t_grad_on: Bool, activates t-component of the lambda derivative
            p: Scalar tensor
            x: Tensor, input to the system
            lam_aug: tuple of tensors of length 3 with
                lam_aug[0] = z-component;
                lam_aug[1] = theta-component;
                lam_aug[2] = t-component.
        """
        grad_lam_z = - jvp(z=lam_aug[0], x=x, v_like_x=self.phi(p, x)) \
                     - vjp(z=self.phi(p, x), x=x, v_like_z=lam_aug[0])

        vjps_wrt_params = []
        for param in self.parameters():
            vjps_wrt_params.append(vjp(z=self.phi(p, x),
                                       x=param,
                                       v_like_z=lam_aug[0]
                                       ).view(-1))

        grad_lam_th = - jvp(z=lam_aug[1], x=x, v_like_x=self.phi(p, x)) \
                      - torch.cat(vjps_wrt_params)
        self.vjp_evals += 2
        self.jvp_evals += 2

        if t_grad_on:
            grad_lam_t = - jvp(z=lam_aug[2], x=x, v_like_x=self.phi(p, x)) \
                         - torch.bmm(jvp(self.phi(p, x), x, self.phi(p, x)),
                                     lam_aug[0],
                                     )
            self.jvp_evals += 2
        else:
            grad_lam_t = torch.zeros(self.batch_dim, 1)

        return grad_lam_z, grad_lam_t, grad_lam_th

    def augmented_grads(self, p_points, x, y, t_grad_on=False):
        """Integrates the p-gradient of the augmented adjoint, Lambda."""
        if not all(p < q for p, q in zip(p_points, p_points[1:])):
            raise ValueError(
                "The list 'p_points' should be strictly increasing.")
        else:
            p_points = torch.tensor(p_points[::-1])

        init_lam_z = vjp(self.cost_fn(self.output(x), y), x)

        init_lam_th = torch.zeros(self.batch_dim,
                                  sum(p.numel() for p in self.parameters()),
                                  requires_grad=True,
                                  ) * x.sum()  # So jvp wrt x is zero, not None

        if t_grad_on:
            init_lam_t = torch.bmm(init_lam_z, self.phi(p_points[0], x))
        else:
            init_lam_t = torch.zeros(self.batch_dim, 1, requires_grad=True)
        init_lam_aug = (init_lam_z, init_lam_th, init_lam_t)

        lam_grad = lambda p, lam_aug: self.lam_aug_dynamics(
            p, x, lam_aug, t_grad_on,
        )

        return odeint(func=lam_grad,
                      y0=init_lam_aug,
                      t=p_points,
                      method=self.integration_method,
                      )

    def forward_dynamics(self, p, x, z):
        """Implements the forward p-gradient of the state z(0; p, x)."""
        self.jvp_evals += 1
        return -jvp(z, x, self.phi(p, x))

    def forward(self, p_points, x):
        """Integrates the p-gradient of the state z(0; p, x) over p_points.

        Args:
            p_points: Points at which to evaluate state z, all less than q.
            x: Tensor, input to the system for any given p in p_points

        Returns:
            odeint: Tensor, first dimension of length(p_points) indexes to
            different the solution z[p] with z[p].shape() = x.shape(). The
            initial condition is located at z(q; q, x) = z[0] = x.
        """

        if not all(p < q for p, q in zip(p_points, p_points[1:])):
            raise ValueError(
                "The list 'p_points' should be strictly increasing.")
        else:
            p_points = torch.tensor(p_points[::-1])
        z_grad = lambda p, z: self.forward_dynamics(p, x, z)
        return odeint(func=z_grad,
                      y0=x,
                      t=p_points,
                      method=self.integration_method,
                      )

    @abc.abstractmethod
    def output(self, zq):
        """The output layer, to reduce feature maps to target dimension."""
        return zq

    @property  # Called upon with each call to self.nfe
    def func_evals(self):
        # Number of evaluations of (phi, vjp, jvp)
        return self.phi.num_evals, self.vjp_evals, self.jvp_evals

    @func_evals.setter  # Called upon with self.func_evals = setter_value
    def func_evals(self, setter_value):
        self.phi.num_evals = setter_value
        self.vjp_evals = setter_value
        self.jvp_evals = setter_value
