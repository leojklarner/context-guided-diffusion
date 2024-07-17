import os

import numpy as np
import torch
from edm.equivariant_diffusion.utils import (
    remove_mean_with_mask,
    assert_correctly_masked,
    assert_mean_zero_with_mask,
)

from analyze.analyze import check_stability
from utils.plotting import plot_graph_of_rings, plot_chain


def rotate_chain(z, n_steps=90):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    theta = np.pi / n_steps
    Qz = torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    ).float()
    Qx = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ]
    ).float()
    Qy = torch.tensor(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps - 1):
        z_x = results[-1][:, :, :3]
        # print(z_x.size(), Q.size())
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        # print(new_x.size())
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_chain_pos(args, model, n_tries, n_nodes=10, std=0.7):
    # helper function for sampling a molecule while saving all the intermediate time steps for visualization
    n_samples = 1

    if args.orientation:
        node_mask = torch.ones(n_samples, 2 * n_nodes, 1).to(args.device)
        edge_mask = 1 - torch.eye(n_nodes)
        edge_mask = torch.cat(
            [
                torch.cat([edge_mask, torch.eye(n_nodes)], dim=1),
                torch.cat([torch.eye(n_nodes), torch.zeros(n_nodes, n_nodes)], dim=1),
            ],
            dim=0,
        ).unsqueeze(0)
        edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(args.device)
        n_nodes *= 2
    else:
        node_mask = torch.ones(n_samples, n_nodes, 1).to(args.device)
        edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
        edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(args.device)

    for i in range(n_tries):
        chain = model.sample_chain(
            n_samples, n_nodes, node_mask, edge_mask, keep_frames=100, std=std
        )
        chain = reverse_tensor(chain)

        x = chain[-1, :, 0:3]
        x_squeeze = x.cpu().detach().numpy()
        node_features = chain[-1, :, 3:]
        node_features = node_features.cpu().detach()
        validity_results = check_stability(
            x_squeeze,
            node_features.argmax(1),
            dataset=args.dataset,
            orientation=args.orientation,
        )
        mol_stable = all(validity_results.values())

        if mol_stable:
            print("Found stable molecule to visualize :)")
            break
        elif i == n_tries - 1:
            print("Did not find stable molecule, showing last sample.")

    x = chain[:, :, :3]
    node_features = chain[:, :, 3:]

    # rotate the output molecule
    n_steps = 90
    x = torch.cat([x, rotate_chain(x[-1:, :, :], n_steps)])
    node_features = torch.cat([node_features, node_features[-1:].repeat(n_steps, 1, 1)])
    return x, node_features


def node2edge_mask(node_mask):
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(
        edge_mask.size(1), dtype=torch.bool, device=node_mask.device
    ).unsqueeze(0)
    edge_mask *= diag_mask
    return edge_mask


def sample_pos_edm(args, model, nodesxsample, std=0.7):
    # helper function for sampling unconditional molecules

    assert int(torch.max(nodesxsample)) <= args.max_nodes
    batch_size = len(nodesxsample)

    # create node and edge masks - according to the number of nodes in each sample
    node_mask = torch.zeros(batch_size, args.max_nodes)
    for i in range(batch_size):
        node_mask[i, 0 : nodesxsample[i]] = 1

    edge_mask = node2edge_mask(node_mask)
    node_mask = node_mask.unsqueeze(2).to(args.device)
    n_nodes = args.max_nodes

    orientation = args.dataset != "cata"
    if orientation:
        node_mask = torch.cat([node_mask, node_mask], dim=1)
        edge_mask = torch.cat(
            [
                torch.cat(
                    [
                        edge_mask,
                        torch.eye(n_nodes).unsqueeze(0).repeat(batch_size, 1, 1),
                    ],
                    dim=1,
                ),
                torch.cat([torch.eye(n_nodes), torch.zeros(n_nodes, n_nodes)], dim=0)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1),
            ],
            dim=2,
        )
        n_nodes *= 2
    edge_mask = edge_mask.view(-1, 1).to(args.device)

    # sample from the EDM model
    x, h = model.sample(batch_size, n_nodes, node_mask, edge_mask, std=std)

    assert_correctly_masked(x, node_mask)
    assert_mean_zero_with_mask(x, node_mask)
    return x, h["categorical"], node_mask, edge_mask


def sample_guidance(args, model, target_function, nodesxsample, scale=1, std=1.0):
    # helper function for sampling conditional molecules - guided by the target function

    # assert int(torch.max(nodesxsample)) <= args.max_nodes
    batch_size = len(nodesxsample)
    max_nodes = nodesxsample.max().item()

    # create node and edge masks - according to the number of nodes in each sample
    node_mask = torch.zeros(batch_size, max_nodes)
    for i in range(batch_size):
        node_mask[i, 0 : nodesxsample[i]] = 1

    # Compute edge_mask
    edge_mask = node2edge_mask(node_mask)
    node_mask = node_mask.unsqueeze(2).to(args.device)

    orientation = args.dataset != "cata"
    if orientation:
        node_mask = torch.cat([node_mask, node_mask], dim=1)
        edge_mask = torch.cat(
            [
                torch.cat(
                    [
                        edge_mask,
                        torch.eye(max_nodes).unsqueeze(0).repeat(batch_size, 1, 1),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [torch.eye(max_nodes), torch.zeros(max_nodes, max_nodes)], dim=0
                )
                .unsqueeze(0)
                .repeat(batch_size, 1, 1),
            ],
            dim=2,
        )
        max_nodes *= 2
    edge_mask = edge_mask.view(-1, 1).to(args.device)

    # sample from the EDM model
    x, h = model.sample_guidance(
        batch_size,
        target_function,
        node_mask,
        edge_mask,
        scale,
        fix_noise=False,
        std=std,
    )

    assert_correctly_masked(x, node_mask)
    assert_mean_zero_with_mask(x, node_mask)
    return x, h["categorical"], node_mask, edge_mask


def save_and_sample_chain_edm(
    args, model, dirname, file_name="chain", n_tries=1, std=0.7
):
    # helper function for sampling and saving a molecule gif
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    try:
        x, one_hot = sample_chain_pos(args, model, n_tries, std=std)
        atom_type = one_hot.argmax(2)
        plot_chain(
            x,
            atom_type,
            dirname=dirname,
            filename=file_name,
            dataset=args.dataset,
        )
    except:
        print("Failed to visualize molecule gif")


def sample_different_sizes_and_save_edm(
    args, model, nodes_dist, prop_dist, n_samples=10, epoch=0, std=0.7
):
    # helper function for sampling and saving a molecules
    n_samples = min(args.batch_size, n_samples)
    nodesxsample = nodes_dist.sample(n_samples)
    try:
        x, one_hot, node_mask = sample_pos_edm(
            args, model, prop_dist, nodesxsample, std=std
        )
        for i in range(n_samples):
            plot_graph_of_rings(
                x[i][node_mask[i, :, 0].bool()],
                one_hot[i][node_mask[i, :, 0].bool()].argmax(1),
                filename=f"{args.exp_dir}/epoch_{epoch}/mol{i}",
                dataset=args.dataset,
                orientation=args.orientation,
            )
    except:
        print("Failed to visualize molecule")
