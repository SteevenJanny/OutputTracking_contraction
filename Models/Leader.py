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


class Leader(nn.Module):
    def __init__(self, system, Hsize=64, layernorm=True):
        super(Leader, self).__init__()
        obs_size = system.observation_size
        ctrl_size = system.input_size
        state_size = system.state_size

        H = Hsize
        self.rnn = nn.GRU(obs_size, H, batch_first=True, num_layers=1)
        self.mlp = MLP(input_size=H + obs_size, output_size=state_size, hidden_size=H, num_layers=1,
                       layer_norm=layernorm)

        self.rnn_control = nn.GRU(state_size + 2 * obs_size, H, batch_first=True, num_layers=1)
        self.mlp_control = MLP(input_size=H + 2 * obs_size, output_size=ctrl_size, hidden_size=H, num_layers=4,
                               layer_norm=layernorm, activation=nn.ReLU())
        self.state_upper_bound, self.state_lower_bound = 100, -100

    def forward(self, observation, system, noise_range=0.0, horizon=200):
        _, h = self.rnn(observation[:, :200])
        h = h[-1]

        initial_state = self.mlp(torch.cat([h, observation[:, 0]], dim=1))
        state_hat, cmd_hat = [initial_state], []
        h_ctrl = None
        for t in range(1, horizon):
            state = state_hat[-1] + (2 * torch.rand_like(state_hat[-1]) - 1) * noise_range
            error = observation[:, t] - system.h(state)

            inpt_ctrl = torch.cat([state, observation[:, t], error], dim=-1)
            _, h_ctrl = self.rnn_control(inpt_ctrl.unsqueeze(1), h_ctrl)
            u = torch.cat([h_ctrl[-1], observation[:, t], error], dim=-1)
            command = self.mlp_control(u)

            next_state = system.simulate(state, command.unsqueeze(-1))

            # At some points, the state diverge and the simulation is not stable anymore
            # to prevent that, we clip the state to a reasonable range
            next_state = next_state * (next_state > self.state_lower_bound).float().detach()
            next_state = next_state * (next_state < self.state_upper_bound).float().detach()

            state_hat.append(next_state)
            cmd_hat.append(command)

        states = torch.stack(state_hat, dim=1)
        observation_hat = system.h(states)
        commands = torch.stack(cmd_hat, dim=1)

        return observation_hat, commands, states
