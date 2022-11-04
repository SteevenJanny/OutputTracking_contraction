import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from Models.controller import Metrics
from Models.Leader import Leader
import argparse
from Dataloader.dynamical_systems import BallAndBeam
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=512)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--leader_source', type=str, default="leader")
parser.add_argument("--train_parameter", type=str, default="true")
parser.add_argument('--name', type=str, default="P_param")  # name of the experiment (save name)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dyn_sys = BallAndBeam(device)


def evaluate():
    np.random.seed(0)
    torch.manual_seed(0)
    model = Metrics(dyn_sys).to(device)

    model.load_state_dict(torch.load(f"trained_models/find_p/{args.name}.pt"))

    leader = Leader(dyn_sys).to(device)
    leader.eval()
    leader.load_state_dict(
        torch.load(f"trained_models/leader/{args.leader_source}.pt", map_location=device))
    reference = torch.load(f"Data/test.pt", map_location=device)

    with torch.no_grad():
        _, psi, x = leader(reference, dyn_sys, 0)
    x = x[:, :-1].reshape(-1, dyn_sys.state_size)
    x.requires_grad = True
    psi = psi.reshape(-1, dyn_sys.input_size, 1)
    dataset = create_data(x, psi)

    test_loader = DataLoader(dataset, batch_size=len(dataset))
    validate(model, test_loader, 0)


def triangulate_P(n, v):
    P = torch.zeros(v.shape[0], n, n).to(device)
    triu = torch.triu_indices(n, n, offset=1)
    P[:, torch.arange(n), torch.arange(n)] = v[:, :n]
    P[:, triu[0], triu[1]] = v[:, n:]
    P[:, triu[1], triu[0]] = v[:, n:]
    return P


def triangulate_dP(n, v):
    dP = torch.zeros(v.shape[0], n, n, n).to(device)
    triu = torch.triu_indices(n, n, offset=1)
    dP[:, torch.arange(n), torch.arange(n), :] = v[:, :n, :]
    dP[:, triu[0], triu[1], :] = v[:, n:, :]
    dP[:, triu[1], triu[0], :] = v[:, n:, :]
    return dP


def get_loss(x, f, g, dfdx, dgdx, actions, jacobian_actions, estimator_params,
             train_parameter_allowed=False):
    e, q, s, lowp = estimator_params
    # Action are triangular part of P
    n = x.shape[-1]
    c = g.shape[-1]
    P = triangulate_P(n, actions)
    dPdx = triangulate_dP(n, jacobian_actions)

    # dPdxg = (dPdx * g.reshape(-1, 1, 1, n)).sum(-1)
    dPdxg = (dPdx.unsqueeze(-1) * g.reshape(-1, 1, 1, n, c)).sum(-2)
    dPdxf = (dPdx * f.reshape(-1, 1, 1, n)).sum(-1)

    dfdx_transpose = dfdx.transpose(1, 2)
    dgdx_transpose = dgdx.transpose(1, 3)

    gT = g.transpose(1, 2)
    I_n = torch.eye(n).unsqueeze(0).repeat(x.shape[0], 1, 1).to(device)

    M1 = dfdx_transpose @ P + P @ dfdx + dPdxf - s * (P @ g @ gT @ P) + q * I_n

    M2 = dgdx_transpose.transpose(1, 2) @ P.unsqueeze(1).repeat(1, c, 1, 1) + \
         P.unsqueeze(1).repeat(1, c, 1, 1) @ dgdx.transpose(1, 2) + \
         dPdxg.transpose(-1, -2).transpose(1, 2)
    M2_up = M2 - e * I_n.unsqueeze(1).repeat(1, c, 1, 1)
    M2_low = - M2 - e * I_n.unsqueeze(1).repeat(1, c, 1, 1)

    M3 = - P + lowp * I_n

    # cost to satisfy conditions
    max_M1 = torch.linalg.eigvalsh(M1).max()
    max_M2_up = torch.linalg.eigvalsh(M2_up).max()
    max_M2_low = torch.linalg.eigvalsh(M2_low).max()
    max_M3 = torch.linalg.eigvalsh(M3).max()

    cost1 = torch.log(F.relu(max_M1) + 1)
    cost2_up = torch.log(F.relu(max_M2_up) + 1)
    cost2_low = torch.log(F.relu(max_M2_low) + 1)
    cost3 = torch.log(F.relu(max_M3) + 1)

    loss_metric = cost1 + cost2_up + cost2_low + cost3
    loss_param = 10 * torch.log(e ** 2 + 1) + torch.log(s ** 2 + 1) - torch.log(q ** 2 + 1) - 0.01 * torch.log(
        lowp ** 2 + 1)

    loss = loss_param if loss_metric == 0 and train_parameter_allowed else loss_metric
    return {'loss': loss, 'loss_metric': loss_metric, "loss_param": loss_param,
            'c1': cost1, 'c2_up': cost2_up, 'c2_low': cost2_low, 'c3': cost3}


def validate(model, loader, epoch):
    costs_list = {'loss': [], 'loss_metric': [], "loss_param": [],
                  'c1': [], 'c2_up': [], 'c2_low': [], 'c3': []}
    model.eval()
    for i, (x, psi, f, g, dfdx, dgdx) in enumerate(loader):
        x = x.to(device)
        psi = psi.to(device)
        f = f.to(device)
        g = g.to(device)
        dfdx = dfdx.to(device)
        dgdx = dgdx.to(device)

        x.requires_grad = True
        actions, estimator_params = model(torch.cat((x, psi.squeeze(-1)), dim=-1))

        jacobian_actions = torch.zeros(x.shape[0], dyn_sys.m_p, dyn_sys.state_size).to(device)
        for j in range(dyn_sys.m_p):
            jacobian_actions[:, j, :] = \
                torch.autograd.grad(actions[:, j].sum(), x, create_graph=True, retain_graph=True)[0]

        costs = get_loss(x=x, f=f, g=g, dfdx=dfdx, dgdx=dgdx,
                         actions=actions, estimator_params=estimator_params,
                         jacobian_actions=jacobian_actions)
        for k, v in costs.items():
            costs_list[k].append(v.item())

    results = {k: np.mean(v) for k, v in costs_list.items()}
    results['epoch'] = epoch
    print("=== EPOCH {} ===".format(epoch + 1))
    print(results)
    return results["loss_metric"]


def generate_dataset_from_leader():
    leader = Leader(dyn_sys).to(device)
    leader.eval()
    leader.load_state_dict(
            torch.load(f"trained_models/leader/{args.leader_source}.pt", map_location=device))
    train_reference = torch.load(f"Data/train.pt").to(device)
    valid_reference = torch.load(f"Data/valid.pt").to(device)

    train_reference = train_reference[::5]
    with torch.no_grad():
        _, psi, x = leader(train_reference, dyn_sys, 0)

    x = x[:, :-1].reshape(-1, dyn_sys.state_size)
    x.requires_grad = True
    psi = psi.reshape(-1, dyn_sys.input_size, 1)
    train_data = create_data(x, psi)

    with torch.no_grad():
        _, psi, x = leader(valid_reference, dyn_sys, 0)
    x = x[:, :-1].reshape(-1, dyn_sys.state_size)
    x.requires_grad = True
    psi = psi.reshape(-1, dyn_sys.input_size, 1)
    valid_data = create_data(x, psi)

    return train_data, valid_data


def create_data(x, psi):
    f = dyn_sys.dynamics(x, psi)
    g = dyn_sys.g(x)
    dfdx = torch.zeros(x.shape[0], dyn_sys.state_size, dyn_sys.state_size).to(device)
    dgdx = torch.zeros(x.shape[0], dyn_sys.state_size, dyn_sys.input_size, dyn_sys.state_size).to(device)
    for i in range(dyn_sys.state_size):
        dfdx[:, i, :] = torch.autograd.grad(f[:, i].sum(), x, retain_graph=True)[0]
        for j in range(dyn_sys.input_size):
            dgdx[:, i, j, :] = torch.autograd.grad(g[:, i, j].sum(), x, retain_graph=True)[0]

    dataset = TensorDataset(x.detach(), psi.detach(), f.detach(), g.detach(), dfdx.detach(), dgdx.detach())
    return dataset


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    name = args.name
    model = Metrics(dyn_sys).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs)

    train_data, test_data = generate_dataset_from_leader()

    train_loader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    memory = torch.inf
    train_parameter_allowed = False
    for epoch in range(args.epochs):
        model.train()
        for i, (x, psi, f, g, dfdx, dgdx) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            psi = psi.to(device)
            f = f.to(device)
            g = g.to(device)
            dfdx = dfdx.to(device)
            dgdx = dgdx.to(device)

            optim.zero_grad()
            x.requires_grad = True  # Track gradient for jacobian

            inpt = torch.cat([x, psi.squeeze(-1)], dim=1)
            actions, estimator_params = model(inpt)
            jacobian_actions = torch.zeros(x.shape[0], dyn_sys.m_p, dyn_sys.state_size).to(device)
            for j in range(dyn_sys.m_p):
                jacobian_actions[:, j, :] = \
                    torch.autograd.grad(actions[:, j].sum(), x, create_graph=True, retain_graph=True)[0]

            costs = get_loss(x=x, f=f, g=g, dfdx=dfdx, dgdx=dgdx,
                             actions=actions, estimator_params=estimator_params,
                             jacobian_actions=jacobian_actions, train_parameter_allowed=train_parameter_allowed)

            costs["loss"].backward()
            optim.step()

        error = validate(model, test_loader, epoch)
        train_parameter_allowed = (error == 0) and (args.train_parameter == "true")
        scheduler.step()
        if error <= memory:
            memory = error
            torch.save(model.state_dict(), f"trained_models/find_p/{name}.pt")
            print("Saved!")


if __name__ == '__main__':
    if args.epochs == 0:
        evaluate()
    else:
        main()
