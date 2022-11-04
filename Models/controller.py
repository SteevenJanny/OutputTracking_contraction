import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, layer_norm=False, activation=nn.Tanh()):
        super(MLP, self).__init__()

        f = []
        if layer_norm:
            f += [nn.LayerNorm(hidden_size)]
        f = [nn.Linear(input_size, hidden_size), ]
        f += [activation]
        for _ in range(num_layers - 1):
            if layer_norm:
                f += [nn.LayerNorm(hidden_size)]
            f += [nn.Linear(hidden_size, hidden_size)]
            f += [activation]

        f += [nn.Linear(hidden_size, output_size)]
        self.f = nn.Sequential(*f)

    def forward(self, x):
        return self.f(x)


class Metrics(nn.Module):
    def __init__(self, dyn_sys, p=(1e-1, 0.5, 30, 0.01), bound=1e-3):
        super(Metrics, self).__init__()
        self.par = nn.Parameter(torch.tensor(p), requires_grad=True)
        self.model = MLP(input_size=dyn_sys.state_size + dyn_sys.input_size,
                         output_size=dyn_sys.m_p,
                         hidden_size=64, num_layers=4,
                         activation=nn.Tanh(),
                         layer_norm=False)
        self.bound = bound

    def get_par(self):
        e, q, s, lowp = torch.abs(self.par)
        lowp = lowp + self.bound
        q = q + self.bound
        return e, q, s, lowp

    def forward(self, x):
        actions = self.model(x)
        e, q, s, lowp = self.get_par()
        return actions, (e, q, s, lowp)
