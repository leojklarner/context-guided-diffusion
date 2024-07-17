import json
import os
import itertools
import random
import pickle
import pandas as pd
from datetime import datetime
from time import time
import warnings
from typing import Tuple
import argparse

import numpy as np
from torch import Tensor, randn
import matplotlib.pyplot as plt

from analyze.analyze import (
    analyze_validity_for_molecules,
    analyze_rdkit_validity_for_molecules,
)
from edm.equivariant_diffusion.utils import (
    assert_correctly_masked,
    assert_mean_zero_with_mask,
)
from models_edm import get_model
from cond_prediction.prediction_args import PredictionArgs
from train_guidance_model import get_cond_predictor_model
from sampling_edm import sample_guidance
from data.aromatic_dataloader import create_data_loaders

from utils.args_edm import Args_EDM
from utils.plotting import plot_graph_of_rings, plot_rdkit
from utils.helpers import switch_grad_off, get_edm_args, get_cond_predictor_args

from torch.nn.functional import softplus

warnings.simplefilter(action="ignore", category=FutureWarning)

import torch


@torch.no_grad()
def predict(model, x, h, node_mask, edge_mask, edm_model) -> Tuple[Tensor, Tensor]:
    bs, n_nodes, n_dims = x.size()
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    node_mask = node_mask.view(bs, n_nodes, 1)

    t = torch.zeros(x.size(0), 1, device=x.device).float()
    x, h, _ = edm_model.normalize(
        x,
        {"categorical": h, "integer": torch.zeros(0).to(x.device)},
        node_mask,
    )
    xh = torch.cat([x, h["categorical"]], dim=-1)
    pred, _ = model(xh, node_mask, edge_mask, t)
    return pred


@torch.no_grad()
def get_target_function_values(x, h, target_function, node_mask, edge_mask, edm_model):
    bs, n_nodes, n_dims = x.size()
    # edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    # edge_mask = edge_mask.repeat(bs, 1, 1).view(-1, 1).to(args.device)
    # node_mask = torch.ones(bs, n_nodes, 1).to(args.device)
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    node_mask = node_mask.view(bs, n_nodes, 1)

    t = torch.zeros(x.size(0), 1, device=x.device).float()
    x, h, _ = edm_model.normalize(
        x,
        {"categorical": h, "integer": torch.zeros(0).to(x.device)},
        node_mask,
    )
    xh = torch.cat([x, h["categorical"]], dim=-1)
    return target_function(xh, node_mask, edge_mask, t)


def eval_stability(x, one_hot, node_mask, edge_mask, dataset="cata"):
    bs, n, _ = x.shape
    atom_type = one_hot.argmax(2).cpu().detach()
    molecule_list = [
        (x[i][node_mask[i, :, 0].bool()].cpu().detach(), atom_type[i])
        for i in range(x.shape[0])
    ]
    stability_dict, molecule_stable_list = analyze_rdkit_validity_for_molecules(
        molecule_list, dataset=dataset
    )
    x = x[stability_dict["molecule_valid_bool"]]
    one_hot = one_hot[stability_dict["molecule_valid_bool"]]
    node_mask = node_mask[stability_dict["molecule_valid_bool"]]
    edge_mask = edge_mask.view(bs, n, n)[stability_dict["molecule_valid_bool"]].view(
        -1, 1
    )
    return stability_dict, x, one_hot, node_mask, edge_mask


def design(args, model, cond_predictor, target_function, scale, n_nodes):
    switch_grad_off([model, cond_predictor])
    model.eval()
    cond_predictor.eval()

    print()
    print("Design molecule...")
    nodesxsample = Tensor([n_nodes] * args.batch_size).long()
    # nodesxsample = nodes_dist.sample(args.batch_size)

    # sample molecules - guidance generation
    start_time = time()
    # sample molecules
    x, one_hot, node_mask, edge_mask = sample_guidance(
        args,
        model,
        target_function,
        nodesxsample,
        scale=scale,
    )
    print(f"Generated {x.shape[0]} molecules in {time() - start_time:.2f} seconds")
    assert_correctly_masked(x, node_mask)
    assert_mean_zero_with_mask(x, node_mask)

    # evaluate stability
    stability_dict, x_stable, atom_type_stable, _, _ = eval_stability(
        x, one_hot, node_mask, edge_mask, dataset=args.dataset
    )
    print(f"{scale=}")
    print(f"{stability_dict['mol_valid']=:.2%} out of {x.shape[0]}")

    # evaluate target function values and prediction
    target_function_values = (
        get_target_function_values(
            x, one_hot, target_function, node_mask, edge_mask, model
        )
        .detach()
        .cpu()
    )
    pred = (
        predict(cond_predictor, x, one_hot, node_mask, edge_mask, model).detach().cpu()
    )

    print(target_function_values.shape)

    print(f"Mean target function value: {target_function_values.mean().item():.4f}")

    if False:
        timestamp = datetime.now().strftime("%m%d_%H:%M:%S")
        dir_name = f"generated_molecules/plots/{args.seed}_{scale}"
        os.makedirs(dir_name, exist_ok=True)

        # find best molecule - can be unvalid
        best_idx = target_function_values.min(0).indices.item()
        best_value = target_function_values[best_idx]
        atom_type = one_hot.argmax(2).cpu().detach()
        print(f"best value: {best_value}, pred: {pred[best_idx]}")
        best_str = ", ".join([f"{t:.3f}" for t in pred[best_idx]])
        plot_graph_of_rings(
            x[best_idx].detach().cpu(),
            atom_type[best_idx].detach().cpu(),
            filename=f"{dir_name}/all.png",
            title=f"{best_str}\n {best_value}",
            dataset=args.dataset,
        )

    # find best valid molecules
    pred = pred[stability_dict["molecule_valid_bool"]]

    target_function_values = target_function_values[
        stability_dict["molecule_valid_bool"]
    ]

    print(
        f"Mean target function value (from valid): {target_function_values.mean().item():.4f}"
    )

    if False:

        best_idxs = target_function_values.argsort()
        for i in range(min(5, target_function_values.shape[0])):
            idx = best_idxs[i]
            value = target_function_values[idx]
            print(f"best value (from stable): {pred[idx]}, " f"score: {value}")

            best_str = ", ".join([f"{t:.3f}" for t in pred[idx]])
            plot_graph_of_rings(
                x_stable[idx].detach().cpu(),
                atom_type_stable[idx].detach().cpu(),
                filename=f"{dir_name}/{i}.pdf",
                title=f"{best_str}\n {value}",
                dataset=args.dataset,
            )
            plot_rdkit(
                x_stable[idx].detach().cpu(),
                atom_type_stable[idx].detach().cpu(),
                filename=f"{dir_name}/mol_{i}.pdf",
                title=f"{best_str}\n {value}",
                dataset=args.dataset,
            )

        # plot target function values histogram
        plt.close()
        plt.hist(target_function_values.numpy().squeeze(), density=True)
        plt.show()

    return stability_dict, x_stable, atom_type_stable, pred, target_function_values


def main(args, cond_predictor_args, scale):
    # Set controllable parameters
    args.batch_size = 512  # number of molecules to generate
    n_nodes = 11  # number of rings in the generated molecules

    (
        train_loader,
        val_loader,
        test_loader,
        context_loader,
        num_node_features,
    ) = create_data_loaders(cond_predictor_args)

    model, _, _ = get_model(args, train_loader, num_node_features)
    cond_predictor = get_cond_predictor_model(
        cond_predictor_args, train_loader.dataset, num_node_features
    )

    # load dataset
    data = pd.read_pickle(
        "/data/stat-cadd/bras5033/guided_diffusion/GaUDI/data/datasets/COMPAS-1x_reduced.csv"
    )
    gap_mean, gap_std = data["GAP_eV"].mean(), data["GAP_eV"].std()
    ip_mean, ip_std = data["aIP_eV"].mean(), data["aIP_eV"].std()
    ea_mean, ea_std = data["aEA_eV"].mean(), data["aEA_eV"].std()

    # define target function - attached two examples for the max gap and OPV target functions.
    # You can create any target function of the predicted properties.
    def target_function_min_gap(_input, _node_mask, _edge_mask, _t):
        pred, _ = cond_predictor(_input, _node_mask, _edge_mask, _t)

        gap = pred[:, 1]
        return gap

    def target_function_opv(_input, _node_mask, _edge_mask, _t):
        pred, _ = cond_predictor(_input, _node_mask, _edge_mask, _t)
        gap = pred[:, 1]
        ip = pred[:, 3]
        ea = pred[:, 4]

        gap = gap * gap_std + gap_mean
        ip = ip * ip_std + ip_mean
        ea = ea * ea_std + ea_mean

        return ip + ea + 3 * gap

    def target_function_opv_mc(_input, _node_mask, _edge_mask, _t):
        pred, _ = cond_predictor(_input, _node_mask, _edge_mask, _t)

        gap_pred_mean = pred[:, 1]
        ip_pred_mean = pred[:, 3]
        ea_pred_mean = pred[:, 4]

        gap_pred_var = softplus(pred[:, 5 + 1])
        ip_pred_var = softplus(pred[:, 5 + 3])
        ea_pred_var = softplus(pred[:, 5 + 4])

        gap = gap_pred_mean - gap_pred_var.sqrt()
        ip = ip_pred_mean - ip_pred_var.sqrt()
        ea = ea_pred_mean - ea_pred_var.sqrt()

        print(gap_pred_mean, gap_pred_var, gap)

        gap = gap * gap_std + gap_mean
        ip = ip * ip_std + ip_mean
        ea = ea * ea_std + ea_mean

        return ip + ea + 3 * gap

    return design(
        args,
        model,
        cond_predictor,
        target_function_opv_mc,
        scale,
        n_nodes,
    )


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--arg_id",
        type=int,
        help="run to generate models for",
    )
    argparser.add_argument(
        "--gen_run_name",
        type=str,
        help="name of the run to generate models for",
    )
    argparser.add_argument(
        "--name_edm_run",
        type=str,
        default="edm_standard_params",
        help="name of the edm run to use",
    )
    argparser.add_argument(
        "--name_cond_predictor_run",
        type=str,
        default="noise_scaled_reg",
        help="name of the cond predictor run to use",
    )
    id_args = argparser.parse_args()

    # create output directory
    os.makedirs(f"generated_molecules/{id_args.gen_run_name}", exist_ok=True)

    # map integer batch job SLURM ID to arguments

    split_types = ["random_split", "cluster_split"]
    context_set_types = ["rings_10", "all"]
    reg_type = ["fseb", "ps", "none"]
    arg_id_list = [
        (s, c, r)
        for s, c, r in itertools.product(split_types, context_set_types, reg_type)
    ]

    # context set does not matter for parameter space regularization
    arg_id_list.remove(("cluster_split", "rings_10", "ps"))
    arg_id_list.remove(("random_split", "rings_10", "ps"))
    arg_id_list.remove(("cluster_split", "rings_10", "none"))
    arg_id_list.remove(("random_split", "rings_10", "none"))

    arg_id_list.append(("cluster_split", "all", "ps_pretrain"))
    arg_id_list.append(("cluster_split", "all", "coral"))
    arg_id_list.append(("cluster_split", "all", "dann"))
    arg_id_list.append(("cluster_split", "all", "domain_confusion"))
    arg_id_list.append(
        ("cluster_split", "all", "fseb")
    )  # manually changed to use model embeds

    split, context_set, reg_type = arg_id_list[id_args.arg_id]

    print("Using arguments:")
    print(f"split: {split}, context_set: {context_set}, reg_type: {reg_type}")

    # load arguments of the pre-trained EDM and conditional predictor models
    args = get_edm_args(
        f"diffusion_training/{id_args.name_edm_run}/run_logs/{id_args.name_edm_run}_{split}"
    )

    results = []

    for s in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
        for rerun_iteration in range(1, 11):

            print(f"\nRunning for scale {s}, rerun {rerun_iteration}")

            # set seeds
            torch.manual_seed(rerun_iteration)
            np.random.seed(rerun_iteration)
            random.seed(rerun_iteration)

            trained_cond_predictor_path = f"standard_regression/{id_args.name_cond_predictor_run}/{id_args.arg_id}_final_{rerun_iteration}.pkl"
            print(
                f"Loading model rerun {rerun_iteration} at {trained_cond_predictor_path}"
            )

            cond_predictor_args = PredictionArgs().parse_args([])

            with open(trained_cond_predictor_path, "rb") as f:
                cond_predictor_args.__dict__ = pickle.load(f)["args"]

            cond_predictor_args.restore = True
            cond_predictor_args.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

            args.seed = rerun_iteration
            cond_predictor_args.seed = rerun_iteration

            print(f"\n\nUsing EDM args: {args}")
            print(f"\n\nUsing cond predictor args: {cond_predictor_args}")

            # Where the magic is
            (
                stability_dict,
                x_stable,
                atom_type_stable,
                pred,
                target_function_values,
            ) = main(args, cond_predictor_args, s)

            results.append(
                {
                    "split": split,
                    "context_set": context_set,
                    "reg_type": reg_type,
                    "scale": s,
                    "rerun_iteration": rerun_iteration,
                    "x_stable": x_stable,
                    "atom_type_stable": atom_type_stable,
                    "pred": pred,
                    "target_function_values": target_function_values,
                    **stability_dict,
                }
            )

    with open(
        f"generated_molecules/{id_args.gen_run_name}/{id_args.arg_id}.pkl", "wb"
    ) as f:
        pickle.dump(results, f)
