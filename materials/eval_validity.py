import json
import math
import random
import warnings

from analyze.analyze import (
    analyze_validity_for_molecules,
    analyze_rdkit_validity_for_molecules,
)
from data.aromatic_dataloader import create_data_loaders
from models_edm import get_model
from sampling_edm import sample_pos_edm, save_and_sample_chain_edm
from utils.args_edm import Args_EDM

from utils.plotting import plot_graph_of_rings
from utils.helpers import get_edm_args

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import torch


def analyze_and_save(args, model, nodes_dist, n_samples=1000, n_chains=1):
    print("-" * 20)
    print("Generate molecules...")

    molecule_list = []
    n_samples = math.ceil(n_samples / args.batch_size) * args.batch_size
    for i in range(n_samples // args.batch_size):
        n_samples = min(args.batch_size, n_samples)
        nodesxsample = nodes_dist.sample(n_samples)
        x, one_hot, node_mask, edge_mask = sample_pos_edm(
            args,
            model,
            nodesxsample,
        )

        x = x.cpu().detach()
        one_hot = one_hot.cpu().detach()
        x = [x[i][node_mask[i, :, 0].bool()] for i in range(x.shape[0])]
        atom_type = [
            one_hot[i][node_mask[i, :, 0].bool()].argmax(dim=1) for i in range(len(x))
        ]
        molecule_list += [(x[i], atom_type[i]) for i in range(len(x))]

    print(f"{len(molecule_list)} molecules generated, starting analysis")

    stability_dict, molecule_stable_list = analyze_validity_for_molecules(
        molecule_list, dataset=args.dataset
    )
    print(f"Stability for {args.exp_dir}")
    for key, value in stability_dict.items():
        try:
            print(f"   {key}: {value:.2%}")
        except:
            pass

    stability_dict, molecule_stable_list = analyze_rdkit_validity_for_molecules(
        molecule_list, dataset=args.dataset
    )
    print(f"RDkit validity for {args.exp_dir}")
    for key, value in stability_dict.items():
        try:
            print(f"   {key}: {value:.2%}")
        except:
            pass

    # plot some molecules
    non_stable_list = list(set(molecule_list) - set(molecule_stable_list))
    if len(non_stable_list) != 0:
        idxs = np.random.randint(0, len(non_stable_list), 5)
        for i in idxs:
            x, atom_type = non_stable_list[i]
            title = f"Non-stable-{i}"
            plot_graph_of_rings(
                x,
                atom_type,
                filename=f"{args.exp_dir}/{title}.png",
                dataset=args.dataset,
            )
    if len(molecule_stable_list) != 0:
        idxs = np.random.randint(0, len(molecule_stable_list), 5)
        for i in idxs:
            x, atom_type = molecule_stable_list[i]
            title = f"Stable-{i}"
            plot_graph_of_rings(
                x,
                atom_type,
                filename=f"{args.exp_dir}/{title}.png",
                dataset=args.dataset,
            )

    # create chains
    for i in range(n_chains):
        save_and_sample_chain_edm(
            args,
            model,
            dirname=args.exp_dir,
            file_name=f"chain{i}",
            n_tries=10,
        )
    return stability_dict


def main(args):
    n_samples = 100
    n_chains = 1

    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(args)

    model, nodes_dist, prop_dist = get_model(args, train_loader)

    # Analyze stability, validity, uniqueness and novelty
    with torch.no_grad():
        analyze_and_save(
            args, model, nodes_dist, n_samples=n_samples, n_chains=n_chains
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args = get_edm_args(
        "/home/tomerweiss/PBHs-design/summary/hetro_l9_c196_orientation2"
    )

    print("Args:", args)

    # Where the magic is
    main(args)
