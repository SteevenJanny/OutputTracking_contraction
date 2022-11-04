import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
from tqdm import tqdm
from Models.Leader import Leader
import matplotlib.pyplot as plt
from Dataloader.dynamical_systems import BallAndBeam
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--window_length", type=int, default=200)
parser.add_argument("--noise_range", type=float, default=0.0)
parser.add_argument("--name", type=str, default="leader")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    system = BallAndBeam(device)
    model = Leader(system).to(device)

    model.load_state_dict(
        torch.load(f"trained_models/leader/{args.name}.pt", map_location=device))
    trajectory = torch.load(f"Data/test.pt", map_location=device)
    model.eval()

    validate(model, trajectory, system, 0, viz=True)


def validate(model, target_trajectory, system, epoch, viz=False):
    cost_list = []
    with torch.no_grad():
        model.eval()
        observation_hat, cmd, _ = model(target_trajectory, system, horizon=args.window_length)
        cost = torch.mean((target_trajectory[:, :args.window_length] - observation_hat) ** 2)
        cost_list.append(cost.item())

        if viz:
            for i in range(observation_hat.shape[0]):
                fig, ax = plt.subplots(1, 1)
                ax.plot(target_trajectory[i].numpy(), label="target")
                ax.plot(observation_hat[i].numpy(), label="prediction")
                ax.legend()
                fig.tight_layout()
                plt.show()

    results = np.mean(cost_list)
    print("=== EPOCH {} ===".format(epoch + 1))
    print(results)
    return results


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    name = args.name
    system = BallAndBeam(device)
    train_references = torch.load(f"Data/train.pt", map_location=device)
    valid_references = torch.load(f"Data/valid.pt", map_location=device)

    train_dataloader = DataLoader(TensorDataset(train_references), batch_size=args.batch_size, shuffle=True)
    model = Leader(system).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    memory = torch.inf

    for epoch in range(args.epochs):
        model.train()
        for i, reference in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            target = reference[0].to(device)
            observation_hat, _, _ = model(target, system, noise_range=args.noise_range, horizon=args.window_length)
            costs = torch.mean((target[:, :args.window_length] - observation_hat) ** 2)

            optimizer.zero_grad()
            costs.backward()
            optimizer.step()

        valid_loss = validate(model, valid_references, system, epoch)
        if valid_loss < memory:
            memory = valid_loss
            torch.save(model.state_dict(), f"trained_models/leader/{name}.pt")
            print("Saved!")


if __name__ == '__main__':
    if args.epochs == 0:
        evaluate()
    else:
        main()
