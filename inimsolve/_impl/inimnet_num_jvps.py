import abc
import torch
import torch.nn as nn
from torch.autograd import grad
from torchdiffeq import odeint


def vjp(z, x, v_like_z=None):
    """Computes the vector-Jacobian product vjp = v^T dot dz/dx."""
    x_copy = x.requires_grad_(True)
    if v_like_z is None:
        v_like_z = torch.ones_like(z).float()
    eval_vjp = grad(z, x_copy, v_like_z, allow_unused=True)
    return eval_vjp[0].requires_grad_(False)


def jvp(z, x, v_like_x=None):
    """Computes the Jacobian-vector product jvp = dz/dx dot v.

    Trick: jvp(z, x, v) = vjp(vjp(z, x, u)^T, u, v))^T for a dummy `u`
    See `https://j-towns.github.io/2017/06/12/A-new-trick.html`
    """
    if v_like_x is None:
        v_like_x = torch.ones_like(x).float()
    v_like_z = torch.ones_like(z).float().requires_grad_(True)
    # No explicit transpose in vjp(z, x, v_like_z) due to vector output
    jvpt = grad(vjp(z, x, v_like_z), v_like_z, v_like_x, allow_unused=True)
    return jvpt[0].requires_grad_(False)


class PhiCnstParams(nn.Module):
    """Controls the constant network parameters with respect to Phi."""

    def __init__(self,
                 dim: int,
                 activation,
                 double_mlp: bool = False,
                 triple_mlp: bool = False,
                 bias_on: bool = True,
                 mult: int = 1,
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


class InImNetNumGrad(nn.Module, metaclass=abc.ABCMeta):
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
                 batch_dim: int,
                 input_dim: int,
                 cost_fn=nn.MSELoss(reduction='none'),
                 activation=nn.ReLU(inplace=True),
                 double_mlp: bool = False,
                 triple_mlp: bool = False,
                 dim_multiplier: int = 1,
                 bias_on: bool = True,
                 integration_method: str = 'euler',
                 grad_shift: float = 1e-5,
                 sym_diff: bool = True
                 ):
        super(InImNetNumGrad, self).__init__()
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
                                 ).requires_grad_(False)
        self.grad_shift = grad_shift
        self.sym_diff = sym_diff
        self.vjp_evals = 0
        self.jvp_evals = 0
        self.jvp_num_evals = 0

    def init_perturbations(self, x):
        """Returns tensor of perturbations of `x`."""
        if len(x.shape) < 2:
            x = x.unsqueeze(0)  # x.shape = [1, xdim]
        x = torch.flatten(x, start_dim=0, end_dim=-2)  # x is 2D
        bdim = x.shape[0]
        xdim = x.shape[1]
        dx = (self.grad_shift * torch.eye(xdim)).repeat([bdim, 1, 1])
        x_plus_dx = x.reshape(bdim, 1, xdim).repeat([1, xdim, 1]) + dx
        x_plus_dx = x_plus_dx.permute(1, 0, 2)
        if self.sym_diff:
            x_minus_dx = x.reshape(bdim, 1, xdim).repeat([1, xdim, 1]) - dx
            x_minus_dx = x_minus_dx.permute(1, 0, 2)
            return torch.cat([x.unsqueeze(0), x_plus_dx, x_minus_dx], dim=0)
        else:
            return torch.cat([x.unsqueeze(0), x_plus_dx], dim=0)

    def jvp_num(self, u, v_like_x=None):
        """Numerically computes the Jacobian-vector product jvp = du/dx dot v.

        Args:
            `u.shape` = [c, b, udim] where c = 2 * xdim + 1 or xdim + 1 is the
            number of perturbed vectors; b is the batch dimension.
            `u[0]` is the original batch of state vectors.
            `u[1:d+1]` is the batch of +ve shifts w.r.t. x.
            `u[d+1:2*d+1]` is the batch of +ve shifts w.r.t. x, if included.
            `delta` is the fixed scalar shift.
        """
        xdim = int((u.shape[0] - 1) / 2 if self.sym_diff else u.shape[0] - 1)
        #print('\n xdim: ', xdim)
        #print(' u.shape: ', u.shape)
        if v_like_x is None:
            v_like_x = torch.ones(u.shape[0], u.shape[1], xdim).float()
        v_shape = v_like_x.shape
        #print(' v_shape: ', v_shape)
        v_like_x = v_like_x.reshape(v_shape[0] * v_shape[1], v_shape[2], 1)
        if self.sym_diff:
            grad_u = (u[1:xdim+1] - u[xdim+1:]) / (2 * self.grad_shift)
            grad_u = grad_u.permute(1, 2, 0).repeat(u.shape[0], 1, 1)
        else:
            grad_u = (u[1:xdim + 1] - u[0].unsqueeze(0).repeat(xdim, 1, 1))
            grad_u /= self.grad_shift
            grad_u = grad_u.permute(1, 2, 0).repeat(u.shape[0], 1, 1)
        return grad_u.bmm(v_like_x).reshape(u.shape)

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

        grad_lam_z = - self.jvp_num(u=lam_aug[0], v_like_x=self.phi(p, x)) \
                     - vjp(z=self.phi(p, x.requires_grad_(True)),
                           x=x,
                           v_like_z=lam_aug[0],
                           )
        grad_lam_z.requires_grad_(False)
        x.requires_grad_(False)

        vjps_wrt_params = []
        for param in self.parameters():
            param.requires_grad_(True)
            vjps_wrt_params.append(vjp(z=self.phi(p, x),
                                       x=param,
                                       v_like_z=lam_aug[0]
                                       ).view(-1))
            param.requires_grad_(False)
        #print('\nvjps_wrt_params: ', len(vjps_wrt_params))
        #for item in vjps_wrt_params: print('loss item: ', item.shape)
        grad_lam_th = - self.jvp_num(u=lam_aug[1], v_like_x=self.phi(p, x)) \
                      - torch.cat(vjps_wrt_params)
        self.vjp_evals += 2
        self.jvp_num_evals += 2

        if t_grad_on:
            grad_lam_t = - self.jvp_num(u=lam_aug[2],
                                        v_like_x=self.phi(p, x)
                                        ) \
                         - torch.bmm(jvp(self.phi(p, x), x, self.phi(p, x)),
                                     lam_aug[0],
                                     )
            self.jvp_evals += 1
            self.jvp_num_evals += 1
        else:
            grad_lam_t = torch.zeros(x.shape[0], self.batch_dim, 1)

        return grad_lam_z, grad_lam_th, grad_lam_t

    def augmented_grads(self, p_points, x, y, t_grad_on=False):
        """Integrates the p-gradient of the augmented adjoint, Lambda."""
        if not all(p < q for p, q in zip(p_points, p_points[1:])):
            raise ValueError(
                "The list 'p_points' should be strictly increasing."
            )
        else:
            p_points = torch.tensor(p_points[::-1])
        x_pturb = self.init_perturbations(x).requires_grad_(True)
        y_pturb = y.repeat(x_pturb.shape[0], 1, 1)
        init_lam_z = vjp(self.cost_fn(self.output(x_pturb), y_pturb), x_pturb)
        self.vjp_evals += 1
        init_lam_z.requires_grad_(False)
        x_pturb.requires_grad_(False)
        init_lam_th = torch.zeros(x_pturb.shape[0],
                                  self.batch_dim,
                                  sum(p.numel() for p in self.parameters()),
                                  )
        if t_grad_on:
            init_lam_t = torch.bmm(init_lam_z, self.phi(p_points[0], x_pturb))
        else:
            init_lam_t = torch.zeros(x_pturb.shape[0], self.batch_dim, 1)
        init_lam_aug = (init_lam_z, init_lam_th, init_lam_t)
        lam_p_grad = lambda p, lam_aug: self.lam_aug_dynamics(
            p, x_pturb, lam_aug, t_grad_on,
        )
        return odeint(func=lam_p_grad,
                      y0=init_lam_aug,
                      t=p_points,
                      method=self.integration_method,
                      )

    def forward_dynamics(self, p, x, z):
        """Implements the forward p-gradient of the state z(0; p, x)."""
        self.jvp_num_evals += 1
        return -self.jvp_num(z, self.phi(p, x))

    def forward(self, p_points, x):
        """Integrates the p-gradient of the state z(0; p, x) over p_points.

        Args:
            p_points: Points at which to evaluate state z, all less than q.
            x: Tensor, input to the system for any given p in p_points

        Returns:
            odeint: Tensor, where the first dimension, of length(p_points),
            indexes z(q; p, x) = z[p][0] with z[p][0].shape() = x.shape(). The
            initial condition is located at z(q; q, x) = z[0][0] = x.
        """

        if not all(p < q for p, q in zip(p_points, p_points[1:])):
            raise ValueError(
                "The list 'p_points' should be strictly increasing.")
        else:
            p_points = torch.tensor(p_points[::-1])
        x_pturb = self.init_perturbations(x)
        return odeint(func=lambda p, z: self.forward_dynamics(p, x_pturb, z),
                      y0=x_pturb,
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
