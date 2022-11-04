import torch


class DynamicalSystem:
    def __init__(self, dt):
        self.dt = dt
        self.observation_size = None
        self.state_size = None
        self.input_size = None

    def simulate(self, x, u):
        assert x.shape[-1] == self.state_size, "State must be of size {}".format(self.state_size)
        assert u.shape[-2] == self.input_size, "Input must be of size {}".format(self.input_size)
        return x + self.dt * self.dynamics(x, u)

    def f(self, x):
        raise NotImplementedError

    def g(self, x):
        raise NotImplementedError

    def h(self, x):
        raise NotImplementedError

    def dynamics(self, x, u):
        return self.f(x) + (self.g(x) @ u).squeeze(-1)

    def generate_target(self, N, batchsize):
        raise NotImplementedError


class BallAndBeam(DynamicalSystem):
    def __init__(self, device):
        super().__init__(dt=1 / 30)
        self.M = 0.05  # Mass of the ball
        self.R = 0.01  # Radius of the ball
        self.J = 0.02  # Moment of inertia of the ball
        self.Jb = 2e-6  # Moment of inertia of the ball
        self.G = 9.81  # Gravitational acceleration
        self.B = self.M / (self.Jb / self.R ** 2 + self.M)

        self.state_size = 4
        self.input_size = 1
        self.observation_size = 1
        self.m_p = 10

        self.x_mean = torch.FloatTensor([-0.0576, 0.0131, 0.0021, -0.0049]).to(device)
        self.x_std = torch.FloatTensor([1.2596, 0.3036, 0.0442, 0.0997]).to(device)

        self.psi_mean = torch.FloatTensor([-0.0050]).to(device)
        self.psi_std = torch.FloatTensor([0.6210]).to(device)

    def f(self, x):
        x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

        x1_dot = x2
        x2_dot = self.B * (x1 * x4 ** 2 - self.G * torch.sin(x3))
        x3_dot = x4
        x4_dot = torch.zeros_like(x4)

        return torch.stack([x1_dot, x2_dot, x3_dot, x4_dot], dim=-1)

    def g(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        g = (x * 0) + torch.FloatTensor([0, 0, 0, 1]).view(1, 4).repeat(x.shape[0], 1).to(x.device)
        return g.reshape(x_shape).unsqueeze(-1)

    def h(self, x):
        return x[..., :self.observation_size]

    def generate_target(self, N, batchsize):
        SIGMA, RHO, BETA = 10, 28, 2.667
        s = [torch.randn(batchsize, 3) * 10, ]  # N(0, 10)
        for t in range(N):
            state = s[-1]
            x, y, z = state[:, 0], state[:, 1], state[:, 2]
            x_dot = SIGMA * (y - x)
            y_dot = RHO * x - y - x * z
            z_dot = x * y - BETA * z
            state_dot = torch.stack([x_dot, y_dot, z_dot], dim=-1)
            s.append(s[-1] + 0.001 * state_dot)

        state = torch.stack(s, dim=1) / 10
        state = state[:, :, :1]
        return state
