import torch.nn as nn

class FixedStepSolver(nn.Module):
    """Fixed-step ODE solver.

    'euler':        Forward Euler Method,
    'midpoint':     Explicit Midpoint Method,
    'rk4':          A Fourth-Order Runge-Kutta Method.
    """

    def __init__(self, step_size: int = 1, method_choice: str = 'euler'):
        self.full_step = step_size
        self.half_step = step_size / 2

        if method_choice == 'euler':
            self.solver = ForwardEuler(step_size)

        elif method_choice == 'midpoint':
            self.solver = ExplicitMidpoint(step_size)

        elif method_choice == 'rk4':
            self.solver = RungeKuttaFour(step_size)

    def forward(self, dzdt, z_in, t_in):
        """Execute a single step of the ODE solver."""
        k1 = dzdt(t_in, z_in)


class ForwardEuler:
    """A first-order ODE solving routine."""

    def __init__(self, step_size):
        self.full_step = step_size

    def step(self, t_in, z_in, dzdt):
        k1 = dzdt(t_in, z_in)
        return z_in + self.full_step * k1


class ExplicitMidpoint:
    """A second-order ODE solving routine."""

    def __init__(self, step_size):
        self.full_step = step_size
        self.half_step = step_size/2

    def step(self, t_in, z_in, dzdt):
        k1 = dzdt(t_in, z_in)
        k2 = dzdt(t_in + self.half_step, z_in + self.half_step * k1)
        return z_in + self.full_step * k2


class RungeKuttaFour:
    """A fourth-order ODE solving routine."""

    def __init__(self, step_size):
        self.full_step = step_size
        self.half_step = step_size/2

    def step(self, t_in, z_in, dzdt):
        k1 = dzdt(t_in, z_in)
        k2 = dzdt(t_in + self.half_step, z_in + self.half_step * k1)
        k3 = dzdt(t_in + self.half_step, z_in + self.half_step * k2)
        k4 = dzdt(t_in + self.full_step, z_in + self.full_step * k3)
        return z_in + (k1 + 2 * (k2 + k3) + k4) * self.full_step / 6
