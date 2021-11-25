import abc
import torch
from torch.autograd import grad
from .ode_solvers import ForwardEuler, ExplicitMidpoint, RungeKuttaFour


INTEGRATION_METHODS = {
    'euler': ForwardEuler,
    'midpoint': ExplicitMidpoint,
    'rk4': RungeKuttaFour
}


def vjp(z, x, v_like_z=None):
    """Computes the vector-Jacobian product vjp = v^T dot dz/dx."""
    if v_like_z == None:
        v_like_z = torch.ones_like(z).float().requires_grad_(True)
    eval_vjp = grad(z, x, v_like_z, create_graph=True, allow_unused=True)
    return eval_vjp[0]


def jvp(z, x, v_like_x=None):
    """Computes the Jacobian-vector product jvp = dz/dx dot v.

    Trick: jvp(z, x, v) = vjp(vjp(z, x, u)^T, u, v))^T for a dummy `u`

    See `https://j-towns.github.io/2017/06/12/A-new-trick.html`
    """
    if v_like_x == None:
        v_like_x = torch.ones_like(x).float().requires_grad_(True)
    v_like_z = torch.ones_like(z).float().requires_grad_(True)
    vjpt = vjp(z, x, v_like_z)  # No explicit transpose due to vector output
    jvpt = grad(vjpt, v_like_z, v_like_x, create_graph=True, allow_unused=True)
    return jvpt[0]


class InImStepSolver(metaclass=abc.ABCMeta):
    """Implements forward and backward steps in InImNet."""

    def __init__(self, method='euler', step_size=1.):
        super(InImStepSolver, self).__init__()
        self.solver = INTEGRATION_METHODS[method](step_size)
        self.vjp_evals = 0
        self.jvp_evals = 0


    @abc.abstractmethod
    def _phi(self, ell, x):
        raise NotImplementedError

    # Compute the gradient functions of z and lam wrt x
    def _grad_z_wrt_ell(self, x):
        self.jvp_evals += 1
        return lambda ell, z: jvp(z, x, self._phi(ell, x))

    def _grad_lam_wrt_ell(self, x):
        self.vjp_evals += 1
        self.jvp_evals += 1
        return lambda ell, lam: jvp(lam, x, self._phi(ell, x)) +\
                                vjp(self._phi(ell, x), x, lam)

    # Perform integration steps
    def z_step(self, ell, x, z_in):
        return self.solver.step(ell, z_in, self._grad_z_wrt_ell(x))

    def lam_step(self, ell, x, lam_in):
        return self.solver.step(ell, lam_in, self._grad_lam_wrt_ell(x))
