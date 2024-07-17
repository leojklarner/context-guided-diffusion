import os
import random
import sys
from pathlib import Path
from time import time
from typing import Tuple

import networkx as nx
import numpy as np
import torch
import pandas as pd
from torch import zeros, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from tqdm import tqdm

from data.mol import Mol, load_xyz, from_rdkit
from data.ring import RINGS_DICT
from utils.args_edm import Args_EDM
from utils.ring_graph import get_rings, get_rings_adj
from utils.molgraph import get_connectivity_matrix, get_edges

# copied in from edm.equivariant_diffusion.utils


def remove_mean_with_mask(x, node_mask):
    if len(node_mask.shape) == 2:
        node_mask = node_mask.unsqueeze(2)
    # assert (x * (1 - node_mask)).abs().sum().item() < 1e-8, (
    #     (x * (1 - node_mask)).abs().sum().item()
    # )
    N = node_mask.sum(1, keepdims=True)
    N = N.clamp(min=1)  # avoid division by zero, won't affect the results

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def assert_correctly_masked(variable, node_mask):
    assert (
        variable * (1 - node_mask)
    ).abs().max().item() < 1e-4, "Variables not masked properly."


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if variable.shape[-1] != 0:
            assert_correctly_masked(variable, node_mask)


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    if rel_error > 1e-2:
        print()
    assert rel_error < 1e-2, f"Mean is not zero, relative_error {rel_error}"


DTYPE = torch.float32
INT_DTYPE = torch.int8
# ATOMS_LIST = __ATOM_LIST__[:8]
ATOMS_LIST = {
    "cata": ["H", "C"],
    "peri": ["H", "C"],
    "hetro": ["H", "C", "B", "N", "O", "S"],
}
RINGS_LIST = {
    "cata": ["Bn"],
    "peri": ["Bn"],
    "hetro": list(RINGS_DICT.keys()) + ["."],
}


def normalize(x, h, node_mask):
    normalize_factors = [3, 4, 10]
    norm_biases = (None, 0.0, 0.0)

    x = x / normalize_factors[0]

    # Casting to float in case h still has long or int type.
    h_cat = (
        (h["categorical"].float() - norm_biases[1]) / normalize_factors[1] * node_mask
    )
    h_int = (h["integer"].float() - norm_biases[2]) / normalize_factors[2]

    # Create new h dictionary.
    h = {"categorical": h_cat, "integer": h_int}

    return x, h


class RandomRotation(object):
    def __call__(self, x):
        M = torch.randn(3, 3)
        Q, __ = torch.linalg.qr(M)
        return x @ Q


class AromaticDataset(Dataset):
    def __init__(self, args, task: str = "train"):
        """
        Args:
            args: All the arguments.
            task: Select the dataset to load from (train/val/test).
        """
        self.csv_file, self.xyz_root = get_paths(args)

        self.task = task
        self.rings_graph = args.rings_graph
        self.normalize = args.normalize
        self.max_nodes = args.max_nodes
        self.return_adj = False
        self.dataset = args.dataset
        self.target_features = getattr(args, "target_features", None)
        self.target_features = (
            self.target_features.split(",") if self.target_features else []
        )
        self.orientation = False if self.dataset == "cata" else True
        self._edge_mask_orientation = None
        self.atoms_list = ATOMS_LIST[self.dataset]
        self.knots_list = RINGS_LIST[self.dataset]

        self.df = getattr(args, f"df_{task}").reset_index()
        self.df = self.df[self.df.n_rings <= args.max_nodes].reset_index()
        if args.normalize:
            train_df = args.df_train
            try:
                target_data = train_df[self.target_features].values
            except:
                self.target_features = [
                    t.replace(" ", "") for t in self.target_features
                ]
                target_data = train_df[self.target_features].values
            self.mean = torch.tensor(target_data.mean(0), dtype=DTYPE)
            self.std = torch.tensor(target_data.std(0), dtype=DTYPE)
        else:
            self.std = torch.ones(1, dtype=DTYPE)
            self.mean = torch.zeros(1, dtype=DTYPE)

        self.examples = np.arange(self.df.shape[0])
        if args.sample_rate < 1:
            random.shuffle(self.examples)
            num_files = round(len(self.examples) * args.sample_rate)
            self.examples = self.examples[:num_files]

        x, node_mask, edge_mask, node_features, y = self.__getitem__(0)[:5]
        self.num_node_features = node_features.shape[1]
        self.num_targets = y.shape[0]

    def get_edge_mask_orientation(self):
        if self._edge_mask_orientation is None:
            self._edge_mask_orientation = torch.zeros(
                2 * self.max_nodes, 2 * self.max_nodes, dtype=torch.bool
            )
            for i in range(self.max_nodes):
                self._edge_mask_orientation[i, self.max_nodes + i] = True
                self._edge_mask_orientation[self.max_nodes + i, i] = True
        return self._edge_mask_orientation.clone()

    def __len__(self):
        return len(self.examples)

    def rescale_loss(self, x):
        # Convert from normalized to the original representation
        if self.normalize:
            x = x * self.std.to(x.device).mean()
        return x

    def get_mol(self, df_row, skip_hydrogen=False) -> Tuple[Mol, list, Tensor, str]:
        name = df_row["molecule"]
        file_path = self.xyz_root + "/" + name
        if os.path.exists(file_path + ".xyz"):
            mol = load_xyz(file_path + ".xyz")
            atom_connectivity = get_connectivity_matrix(
                mol.atoms, skip_hydrogen=skip_hydrogen
            )  # build connectivity matrix
            # edges = bonds
        elif os.path.exists(file_path + ".pkl"):
            mol, atom_connectivity = from_rdkit(file_path + ".pkl")
        else:
            raise NotImplementedError(file_path)
        edges = get_edges(atom_connectivity)
        return mol, edges, atom_connectivity, name

    def get_rings(self, df_row):
        name = df_row["molecule"]
        os.makedirs(self.xyz_root + "_rings_preprocessed", exist_ok=True)
        preprocessed_path = self.xyz_root + "_rings_preprocessed/" + name + ".xyz"
        if Path(preprocessed_path).is_file():
            x, adj, node_features, orientation = torch.load(preprocessed_path)
        else:
            mol, edges, atom_connectivity, _ = self.get_mol(df_row, skip_hydrogen=True)
            # get_figure(mol, edges, showPlot=True, filename='4.png')
            mol_graph = nx.Graph(edges)
            knots = get_rings(mol.atoms, mol_graph)
            adj = get_rings_adj(knots)
            x = torch.tensor([k.get_coord() for k in knots], dtype=DTYPE)
            knot_type = torch.tensor(
                [self.knots_list.index(k.cycle_type) for k in knots]
            ).unsqueeze(1)
            node_features = (
                one_hot(knot_type, num_classes=len(self.knots_list)).squeeze(1).float()
            )
            orientation = [k.orientation for k in knots]
            torch.save([x, adj, node_features, orientation], preprocessed_path)
        return x, adj, node_features, orientation

    def get_atoms(self, df_row):
        name = df_row["molecule"]
        preprocessed_path = self.xyz_root + "_atoms_preprocessed/" + name + ".xyz"
        if Path(preprocessed_path).is_file():
            x, adj, node_features = torch.load(preprocessed_path)
        else:
            mol, edges, atom_connectivity, _ = self.get_mol(df_row)
            # get_figure(mol, edges, showPlot=True)
            x = torch.tensor([a.get_coord() for a in mol.atoms], dtype=DTYPE)
            atom_element = torch.tensor(
                [self.atoms_list.index(atom.element) for atom in mol.atoms]
            ).unsqueeze(1)
            node_features = (
                one_hot(atom_element, num_classes=len(self.atoms_list))
                .squeeze(1)
                .float()
            )
            adj = atom_connectivity
            torch.save([x, adj, node_features], preprocessed_path)
        return x, adj, node_features

    def get_all(self, df_row):
        # extract targets
        y = torch.tensor(
            df_row[self.target_features].values.astype(np.float32), dtype=DTYPE
        )
        if self.normalize:
            y = (y - self.mean) / self.std

        # creation of nodes, edges and there features
        x, adj, node_features, orientation = self.get_rings(df_row)

        if self.orientation:
            # adjust to max nodes shape
            n_nodes = x.shape[0]
            x_r = torch.tensor([random.sample(o, 1)[0] for o in orientation])
            x_full = zeros(self.max_nodes * 2, 3)
            x_full[:n_nodes] = x
            x_full[self.max_nodes : self.max_nodes + n_nodes] = x_r

            node_mask = zeros(self.max_nodes * 2)
            node_mask[:n_nodes] = 1
            node_mask[self.max_nodes : self.max_nodes + n_nodes] = 1

            node_features_full = zeros(self.max_nodes * 2, node_features.shape[1])
            node_features_full[:n_nodes, :] = node_features
            # mark the orientation nodes as additional ring type
            node_features_full[self.max_nodes : self.max_nodes + n_nodes, -1] = 1

            edge_mask_tmp = node_mask[: self.max_nodes].unsqueeze(0) * node_mask[
                : self.max_nodes
            ].unsqueeze(1)
            # mask diagonal
            diag_mask = ~torch.eye(self.max_nodes, dtype=torch.bool)
            edge_mask_tmp *= diag_mask
            edge_mask = self.get_edge_mask_orientation()
            edge_mask[: self.max_nodes, : self.max_nodes] = edge_mask_tmp

            if self.return_adj:
                adj_full = self.get_edge_mask_orientation()
                adj_full[:n_nodes, :n_nodes] = adj
        else:
            # adjust to max nodes shape
            n_nodes = x.shape[0]
            x_full = zeros(self.max_nodes, 3)

            node_mask = zeros(self.max_nodes)
            x_full[:n_nodes] = x
            node_mask[:n_nodes] = 1

            node_features_full = zeros(self.max_nodes, node_features.shape[1])
            node_features_full[:n_nodes, :] = node_features
            # node_features_full = zeros(self.max_nodes, 0)

            # edge_mask = zeros(self.max_nodes, self.max_nodes)
            # edge_mask[:n_nodes, :n_nodes] = adj
            # edge_mask = edge_mask.view(-1, 1)

            edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
            # mask diagonal
            diag_mask = ~torch.eye(self.max_nodes, dtype=torch.bool)
            edge_mask *= diag_mask
            # edge_mask = edge_mask.view(-1, 1)

            if self.return_adj:
                adj_full = zeros(self.max_nodes, self.max_nodes)
                adj_full[:n_nodes, :n_nodes] = adj

        if self.return_adj:
            return x_full, node_mask, edge_mask, node_features_full, adj_full, y
        else:
            return x_full, node_mask, edge_mask, node_features_full, y

    def __getitem__(self, idx):
        index = self.examples[idx]
        df_row = self.df.loc[index]
        return self.get_all(df_row)


def get_paths(args):
    if not hasattr(args, "dataset"):
        csv_path = args.csv_file
        xyz_path = args.xyz_root
    elif args.dataset == "cata":
        csv_path = "/data/stat-cadd/bras5033/guided_diffusion/GaUDI/data/datasets/COMPAS-1x_reduced.csv"
        xyz_path = "/data/stat-cadd/bras5033/guided_diffusion/GaUDI/data/datasets/pahs-cata-34072-xyz"
    elif args.dataset == "peri":
        csv_path = "/home/tomerweiss/PBHs-design/data/peri-xtb-data-55821.csv"
        xyz_path = "/home/tomerweiss/PBHs-design/data/peri-cata-89893-xyz"
    elif args.dataset == "hetro":
        csv_path = "/home/tomerweiss/PBHs-design/data/db-474K-OPV-filtered.csv"
        xyz_path = "/home/tomerweiss/PBHs-design/data/db-474K-xyz"
    elif args.dataset == "hetro-dft":
        csv_path = "/home/tomerweiss/PBHs-design/data/db-15067-dft.csv"
        xyz_path = ""
    else:
        raise NotImplementedError
    return csv_path, xyz_path


def get_splits(args, random_seed=42, val_frac=0.1, test_frac=0.1):
    # this seed reset was in original code,
    # bit messy but will be kept for reproducibility
    np.random.seed(seed=random_seed)
    csv_path, _ = get_paths(args)
    if hasattr(args, "dataset") and args.dataset == "hetro":
        targets = (
            args.target_features.split(",")
            if getattr(args, "target_features", None) is not None
            else []
        )
        df = pd.read_csv(csv_path, usecols=["name", "nRings", "inchi"] + targets)
        df.rename(columns={"nRings": "n_rings", "name": "molecule"}, inplace=True)
        args.max_nodes = min(args.max_nodes, 10)
    else:
        df = pd.read_pickle(csv_path)

    if args.context_set == "all":
        df_context = df.copy()
    elif args.context_set == "rings_10":
        df_context = df[df[args.split] == -1]
    else:
        raise NotImplementedError(f"Unknown context set {args.context_set}")

    df_train = df[df[args.split] == 0]
    df_val = df[df[args.split] == 1]
    df_test = df[df[args.split] == 2]

    # df_test = df.sample(frac=test_frac, random_state=random_seed)
    # df = df.drop(df_test.index)
    # df_val = df.sample(frac=val_frac, random_state=random_seed)
    # df_train = df.drop(df_val.index)

    return df_train, df_val, df_test, df_context


def get_split_indices(args):

    csv_path, _ = get_paths(args)
    df = pd.read_pickle(csv_path)

    train_indices = df[df[args.split] == 0].index.values
    val_indices = df[df[args.split] == 1].index.values
    test_indices = df[df[args.split] == 2].index.values

    if args.context_set == "all":
        print("Using all molecules for context set")
        context_indices = df[df[args.split] != 0].index.values
    elif args.context_set == "rings_10":
        print("Using molecules with 10 rings or fewer for context set")
        context_indices = df[df[args.split] == -1].index.values
    else:
        raise NotImplementedError(f"Unknown context set {args.context_set}")

    return df, train_indices, val_indices, test_indices, context_indices


def create_data_loaders(args):

    _, xyz_root = get_paths(args)

    (
        df_full,
        train_indices,
        val_indices,
        test_indices,
        context_indices,
    ) = get_split_indices(args)

    # use provided data loader to process full dataset
    # needs to be called args.df_train for hard-coded normalization
    args.df_full = df_full
    full_dataset = AromaticDataset(
        args=args,
        task="full",
    )

    # cache processing and concatenate tensors

    cache_file = os.path.join(xyz_root, "cached_dataset.pt")

    if os.path.exists(cache_file):

        print("Loading cached dataset ... ", end="")
        xhs, node_masks, edge_masks, ys = torch.load(cache_file)
        print("Done!")

    else:

        print("Caching full dataset ... ")
        xs, node_masks, edge_masks, node_features, ys = [], [], [], [], []

        for i in range(len(df_full)):
            x, node_mask, edge_mask, node_feature, y = full_dataset[i]
            xs.append(x)
            node_masks.append(node_mask)
            edge_masks.append(edge_mask)
            node_features.append(node_feature)
            ys.append(y)

        xs = torch.stack(xs)
        node_masks = torch.stack(node_masks).unsqueeze(2)
        edge_masks = torch.stack(edge_masks)
        node_features = torch.stack(node_features)
        ys = torch.stack(ys)

        # apply location and node feature preprocessing and normalization

        xs = remove_mean_with_mask(xs, node_masks)
        check_mask_correct([xs, node_features], node_masks)
        assert_mean_zero_with_mask(xs, node_masks)

        print("Normalizing node features ... ", end="")
        xs, node_features = normalize(
            xs,
            {"categorical": node_features, "integer": torch.zeros(0).to(xs.device)},
            node_masks,
        )
        xhs = torch.cat([xs, node_features["categorical"]], dim=-1)
        bs, n_nodes, n_dims = xs.size()
        assert_correctly_masked(xs, node_masks)
        edge_masks = edge_masks.view(bs, n_nodes * n_nodes)

        print("Done!")

        torch.save(
            (xhs, node_masks, edge_masks, ys),
            cache_file,
        )
        print("Caching done!")

    print(xhs.shape, node_masks.shape, edge_masks.shape, ys.shape)

    num_node_features = xhs.shape[-1] - 3  # subtract 3 for x,y,z coords

    # move data to device

    print("Moving data to device ... ", end="")

    xhs = xhs.to(args.device)
    node_masks = node_masks.to(args.device)
    edge_masks = edge_masks.to(args.device)
    ys = ys.to(args.device)

    print("Done!")

    # normalize targets

    print("Normalizing targets ... ", end="")
    ys_mean = ys[train_indices].mean(0)
    ys_std = ys[train_indices].std(0)
    ys = (ys - ys_mean) / ys_std
    print("Done!")

    # create train/val/test/context datasets

    train_dataset = torch.utils.data.TensorDataset(
        xhs[train_indices],
        node_masks[train_indices],
        edge_masks[train_indices],
        ys[train_indices],
    )

    val_dataset = torch.utils.data.TensorDataset(
        xhs[val_indices],
        node_masks[val_indices],
        edge_masks[val_indices],
        ys[val_indices],
    )

    test_dataset = torch.utils.data.TensorDataset(
        xhs[test_indices],
        node_masks[test_indices],
        edge_masks[test_indices],
        ys[test_indices],
    )

    context_dataset = torch.utils.data.TensorDataset(
        xhs[context_indices],
        node_masks[context_indices],
        edge_masks[context_indices],
        ys[context_indices],
    )

    # create data loaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_indices),
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_indices),
        shuffle=False,
    )

    context_loader = DataLoader(
        context_dataset,
        batch_size=args.n_context_points,
        shuffle=True,
    )

    return train_loader, val_loader, test_loader, context_loader, num_node_features
