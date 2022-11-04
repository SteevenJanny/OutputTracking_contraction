import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
from Dataloader.dynamical_systems import BallAndBeam
from Models.controller import Metrics, MLP
from Models.Leader import Leader

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=512)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--metric_name', type=str, default="P_param")
parser.add_argument('--leader_source', type=str, default="leader")
parser.add_argument('--name', type=str, default="alpha_param")  # name of the experiment (save name)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dyn_sys = BallAndBeam(device)


def evaluate():
    raise NotImplementedError


def triangulate_P(n, v):
    P = torch.zeros(v.shape[0], n, n).to(device)
    triu = torch.triu_indices(n, n, offset=1)
    P[:, torch.arange(n), torch.arange(n)] = v[:, :n]
    P[:, triu[0], triu[1]] = v[:, n:]
    P[:, triu[1], triu[0]] = v[:, n:]
    return P


def generate_dataset(metric):
    # learning parameters
    leader = Leader(dyn_sys).to(device)
    leader.load_state_dict(
        torch.load(f"trained_models/leader/{args.leader_source}.pt", map_location=device))
    train_reference = torch.load(f"Data/train.pt").to(device)
    valid_reference = torch.load(f"Data/valid.pt").to(device)

    leader.eval()
    train_reference = train_reference[::5]
    with torch.no_grad():
        _, psi, x = leader(train_reference, dyn_sys, 0)
        x = x[:, :-1].reshape(-1, dyn_sys.state_size)
        psi = psi.reshape(-1, dyn_sys.input_size, 1)
        train_dataset = create_data(x, psi, metric)

        _, psi, x = leader(valid_reference, dyn_sys, 0)
        x = x[:, :-1].reshape(-1, dyn_sys.state_size)
        psi = psi.reshape(-1, dyn_sys.input_size, 1)

        valid_dataset = create_data(x, psi, metric)
    return train_dataset, valid_dataset


def create_data(x, psi, metric):
    g = dyn_sys.g(x)

    aug_state = torch.cat((x, psi.squeeze(-1)), dim=-1)
    actions, _ = metric(aug_state)
    P = triangulate_P(dyn_sys.state_size, actions)
    gT = g.transpose(-1, -2)
    gTP = gT @ P

    return TensorDataset(x, psi, gTP)


def validate(policy, loader, epoch=0):
    cost_list = []
    for i, (x, psi, dadx) in enumerate(loader):
        x = x.to(device)
        psi = psi.to(device)
        dadx = dadx.to(device)

        x.requires_grad = True

        aug_state = torch.cat((x, psi.squeeze(-1)), dim=1).to(device)
        alpha = policy(aug_state)

        jac = torch.zeros(x.shape[0], dyn_sys.input_size, dyn_sys.state_size).to(device)
        for j in range(dyn_sys.input_size):
            jac[:, j, :] = \
                torch.autograd.grad(alpha[:, j].sum(), x, create_graph=True, retain_graph=True)[0]

        loss = torch.mean((jac - dadx) ** 2)
        cost_list.append(loss.item())
    results = np.mean(cost_list)
    print("=== EPOCH {} ===".format(epoch + 1))
    print(results)
    return results


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    name = args.name

    policy = MLP(input_size=dyn_sys.state_size + dyn_sys.input_size,
                 output_size=dyn_sys.input_size,
                 hidden_size=64, num_layers=4).to(device)

    metric = Metrics(dyn_sys).to(device)
    metric.load_state_dict(torch.load(f"trained_models/find_p/{args.metric_name}.pt"))
    metric.eval()

    net_optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(net_optimizer, args.epochs)

    # generate dataset
    train_data, test_data = generate_dataset(metric)
    train_loader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    memory = torch.inf
    for epoch in range(args.epochs):
        policy.train()
        for i, (x, psi, dadx) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            psi = psi.to(device)
            dadx = dadx.to(device)

            x.requires_grad = True

            aug_state = torch.cat((x, psi.squeeze(-1)), dim=1).to(device)
            alpha = policy(aug_state)

            jac = torch.zeros(x.shape[0], dyn_sys.input_size, dyn_sys.state_size).to(device)
            for j in range(dyn_sys.input_size):
                jac[:, j, :] = \
                    torch.autograd.grad(alpha[:, j].sum(), x, create_graph=True, retain_graph=True)[0]

            loss = torch.mean((jac - dadx) ** 2)

            net_optimizer.zero_grad()
            loss.backward()
            net_optimizer.step()

        error = validate(policy, test_loader, epoch)
        scheduler.step()
        if error < memory:
            memory = error
            torch.save(policy.state_dict(), f"trained_models/find_alpha/{name}.pt")
            print("Saved!")


if __name__ == '__main__':
    if args.epochs == 0:
        evaluate()
    else:
        main()
