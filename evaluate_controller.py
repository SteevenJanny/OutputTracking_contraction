import torch
import argparse
from Models.Leader import Leader
import matplotlib.pyplot as plt
import wandb
from Dataloader.dynamical_systems import BallAndBeam
from Models.controller import MLP

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--window_length", type=int, default=600)
parser.add_argument("--leader_name", type=str, default="leader")
parser.add_argument("--alpha_name", type=str, default="alpha_param")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYSTEMS = BallAndBeam(device)

plt.style.use("seaborn")


def evaluate():
    wandb.init(project="LeaderContraction", entity='sj777', config={}, mode="disabled")

    system = BallAndBeam(device)
    model = Leader(system).to(device)
    alpha = MLP(input_size=system.state_size + system.input_size,
                output_size=system.input_size,
                hidden_size=64, num_layers=4).to(device)

    model.load_state_dict(torch.load(f"trained_models/leader/{args.leader_name}.pt", map_location=device))
    alpha.load_state_dict(torch.load(f"trained_models/find_alpha/{args.alpha_name}.pt", map_location=device))
    target_trajectory = torch.load(f"Data/test.pt")

    model.eval()
    alpha.eval()
    target_trajectory = target_trajectory[:, :args.window_length, :].to(device)
    with torch.no_grad():
        model.eval()
        observation_hat, psi, pi = model(target_trajectory, system, noise_range=0, horizon=args.window_length)
        K = 2
        x0 = pi[:, 0].clone() + torch.randn_like(pi[:, 0]) * 0.1
        x = [x0]  # initial state
        cmd = []
        for t in range(1, observation_hat.shape[1]):
            inpt_x = x[-1]
            inpt_psi = psi[:, t - 1]
            inpt_pi = pi[:, t - 1]
            beta_x = alpha(torch.cat([inpt_x, inpt_psi], dim=-1))
            beta_pi = alpha(torch.cat([inpt_pi, inpt_psi], dim=-1))

            u = psi[:, t - 1] - K * (beta_x - beta_pi)
            cmd.append(u)
            next_x = system.simulate(x[-1], u.unsqueeze(-1))
            x.append(next_x)
        x = torch.stack(x, dim=1)
        for b in range(len(x)):
            fig, ax = plt.subplots(4, 1)
            for i in range(4):
                ax[i].plot(pi[b, :, i].cpu().numpy(), label="leader")
                ax[i].plot(x[b, :, i].cpu().numpy(), label="follower")
                ax[i].legend()

            plt.show()


if __name__ == '__main__':
    evaluate()
