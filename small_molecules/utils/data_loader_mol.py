import os
import pickle
from time import time
import numpy as np
import networkx as nx
import pandas as pd

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import json


def load_mol(filepath):
    print(f"Loading file {filepath}")
    if not os.path.exists(filepath):
        raise ValueError(f"Invalid filepath {filepath} for dataset")
    load_data = np.load(filepath)
    result = []
    i = 0
    while True:
        key = f"arr_{i}"
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    # convert a tuple of lists to a list of tuples
    return list(map(lambda x, a: (x, a), result[0], result[1]))


class MolDataset(Dataset):
    def __init__(self, mols, transform=None):
        self.mols = mols
        self.transform = transform

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        if self.transform is None:
            return self.mols[idx]
        else:
            return self.transform(self.mols[idx])


class MolDataset_prop(Dataset):
    def __init__(self, mols, prop, idx, transform=None):
        self.mols = mols
        self.transform = transform
        df = pd.read_csv("data/zinc250k.csv").iloc[idx]

        if "parp1" in prop:
            protein = "parp1"
        elif "fa7" in prop:
            protein = "fa7"
        elif "5ht1b" in prop:
            protein = "5ht1b"
        elif "jak2" in prop:
            protein = "jak2"
        elif "braf" in prop:
            protein = "braf"

        self.y = np.clip(df[protein], 0.0, 20.0) / 20.0
        if "qed" in prop:
            self.y *= df["qed"]
        if "sa" in prop:
            self.y *= df["sa"]

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        if self.transform is None:
            return self.mols[idx], self.y.iloc[idx]
        else:
            return (*self.transform(self.mols[idx]), self.y.iloc[idx])


def dataloader(config, get_graph_list=False, prop=None, device=None):

    print("Initial prop passed to dataloader:", prop)

    # remove "retrain_i" indicator from property name
    if prop is not None:
        for i in range(10):
            if f"retrain_{i}" in prop:
                prop = prop.replace(f"_retrain_{i}", "")

    print("Actual prop used by:", prop)

    use_hard_split = True

    dataset_cache_file = os.path.join(
        "/data/stat-cadd/bras5033/guided_diffusion/MOOD/data/cached_dataloaders",
        f"dataset_{config.data.data.lower()}_{prop}_{get_graph_list}_{use_hard_split}.pkl",
    )

    if os.path.exists(dataset_cache_file):
        start_time = time()
        print(f"Loading cached datasets from {dataset_cache_file}")
        with open(dataset_cache_file, "rb") as f:
            train_dataset, test_dataset = pickle.load(f)

    else:

        start_time = time()

        if config.data.data == "QM9":

            def transform_RGCN(data):
                x, adj = data
                # the last place is for virtual nodes
                # 6: C, 7: N, 8: O, 9: F
                x_ = np.zeros((9, 5))
                indices = np.where(x >= 6, x - 6, 4)
                x_[np.arange(9), indices] = 1
                x = torch.tensor(x_).to(torch.float32)
                # single, double, triple and no-bond; the last channel is for virtual edges
                adj = np.concatenate(
                    [adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0
                ).astype(np.float32)
                return x, adj  # (9, 5), (4, 9, 9)

        elif config.data.data == "ZINC250k":

            def transform_RGCN(data):
                x, adj = data
                # the last place is for virtual nodes
                # 6: C, 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
                zinc250k_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
                x_ = np.zeros((38, 10), dtype=np.float32)
                for i in range(38):
                    ind = zinc250k_atomic_num_list.index(x[i])
                    x_[i, ind] = 1.0
                x = torch.tensor(x_).to(torch.float32)
                # single, double, triple and no-bond; the last channel is for virtual edges
                adj = np.concatenate(
                    [adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0
                ).astype(np.float32)
                return x, adj  # (38, 10), (4, 38, 38)

        def transform_GCN(data):
            x, adj = transform_RGCN(data)
            x = x[:, :-1]
            adj = torch.tensor(adj.argmax(axis=0))
            # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            return x, adj

        mols = load_mol(
            os.path.join(config.data.dir, f"{config.data.data.lower()}_kekulized.npz")
        )

        if use_hard_split:
            valid_file = os.path.join(
                config.data.dir,
                f'valid_idx_{config.data.data.lower()}_{config.train.prop.split("_")[0]}.json',
            )

        else:
            valid_file = os.path.join(
                config.data.dir,
                f"valid_idx_{config.data.data.lower()}.json",
            )

        with open(valid_file) as f:
            test_idx = json.load(f)
            print(f"Loading file {valid_file}")

        if config.data.data == "QM9":
            test_idx = test_idx["valid_idxs"]
            test_idx = [int(i) for i in test_idx]

        train_idx = [i for i in range(len(mols)) if i not in test_idx]

        train_mols = [mols[i] for i in train_idx]
        test_mols = [mols[i] for i in test_idx]

        print(
            f"Number of training mols: {len(train_idx)} | Number of test mols: {len(test_idx)}"
        )

        # preprocess the dataset
        train_x, train_adj = [], []
        for mol in tqdm(train_mols, desc="Preprocessing training molecules"):
            x, adj = transform_GCN(mol)
            train_x.append(x)
            train_adj.append(adj)

        train_x, train_adj = torch.stack(train_x), torch.stack(train_adj)
        print(f"Training set: {train_x.shape}, {train_adj.shape}")

        test_x, test_adj = [], []
        for mol in tqdm(test_mols, desc="Preprocessing test molecules"):
            x, adj = transform_GCN(mol)
            test_x.append(x)
            test_adj.append(adj)

        test_x, test_adj = torch.stack(test_x), torch.stack(test_adj)
        print(f"Test set: {test_x.shape}, {test_adj.shape}")

        if prop is None:
            train_dataset = TensorDataset(train_x, train_adj)
            test_dataset = TensorDataset(test_x, test_adj)
        else:

            # get labels
            df = pd.read_csv("data/zinc250k.csv")

            if "parp1" in prop:
                protein = "parp1"
            elif "fa7" in prop:
                protein = "fa7"
            elif "5ht1b" in prop:
                protein = "5ht1b"
            elif "jak2" in prop:
                protein = "jak2"
            elif "braf" in prop:
                protein = "braf"
            else:
                raise ValueError(f"Invalid property {prop}")

            y = np.clip(df[protein], 0.0, 20.0) / 20.0
            if "qed" in prop:
                y *= df["qed"]
            if "sa" in prop:
                y *= df["sa"]

            y = torch.Tensor(y).to(torch.float32)
            print(f"Labels: {y.shape}")

            train_dataset = TensorDataset(train_x, train_adj, y[train_idx])
            test_dataset = TensorDataset(test_x, test_adj, y[test_idx])

        print(f"Caching datasets to {dataset_cache_file}")
        with open(dataset_cache_file, "wb") as f:
            pickle.dump((train_dataset, test_dataset), f)

    if get_graph_list:
        train_dataloader = [
            nx.from_numpy_matrix(np.array(adj)) for x, adj in train_dataset
        ]
        test_dataloader = [
            nx.from_numpy_matrix(np.array(adj)) for x, adj in test_dataset
        ]

    else:

        # push datasets to device
        train_dataset.tensors = tuple(
            tensor.to(f"cuda:{device[0]}") for tensor in train_dataset.tensors
        )
        test_dataset.tensors = tuple(
            tensor.to(f"cuda:{device[0]}") for tensor in test_dataset.tensors
        )

        for tensor in train_dataset.tensors:
            print(tensor.shape, tensor.dtype, tensor.device)

        train_dataloader = DataLoader(
            train_dataset, batch_size=config.data.batch_size, shuffle=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=config.data.batch_size, shuffle=True
        )

    print(f"{time() - start_time:.2f} sec elapsed for data loading")

    return train_dataloader, test_dataloader


def contextloader(config, get_graph_list=False, device=None):

    contextset_cache_file = os.path.join(
        "/data/stat-cadd/bras5033/guided_diffusion/MOOD/data/cached_contextloaders",
        f"contextset_{config.data.context.lower()}_{get_graph_list}.pkl",
    )

    if os.path.exists(contextset_cache_file):
        start_time = time()
        print(f"Loading cached contextset from {contextset_cache_file}")
        with open(contextset_cache_file, "rb") as f:
            context_dataset = pickle.load(f)

    else:

        start_time = time()

        if config.data.context in ["ZINC500k", "ZINC50k", "ZINC5k", "ZINC05k", "custom_qm9", "qmugs", "ZINC50k_most_similar", "ZINC50k_least_similar"]:

            def transform_RGCN(data):
                x, adj = data
                # the last place is for virtual nodes
                # 6: C, 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
                zinc250k_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
                x_ = np.zeros((38, 10), dtype=np.float32)
                for i in range(38):
                    ind = zinc250k_atomic_num_list.index(x[i])
                    x_[i, ind] = 1.0
                x = torch.tensor(x_).to(torch.float32)
                # single, double, triple and no-bond; the last channel is for virtual edges
                adj = np.concatenate(
                    [adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0
                ).astype(np.float32)
                return x, adj  # (38, 10), (4, 38, 38)

        else:
            raise ValueError(f"Invalid context set {config.data.context}")

        def transform_GCN(data):
            x, adj = transform_RGCN(data)
            x = x[:, :-1]
            adj = torch.tensor(adj.argmax(axis=0))
            # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            return x, adj

        context_mols = load_mol(
            os.path.join(
                config.data.dir, f"{config.data.context.lower()}_kekulized.npz"
            )
        )

        print(f"Number of context set molecules: {len(context_mols)}")

        # preprocess the dataset
        context_x, context_adj = [], []
        for mol in tqdm(context_mols, desc="Preprocessing context set molecules"):
            x, adj = transform_GCN(mol)
            context_x.append(x)
            context_adj.append(adj)

        context_x, context_adj = torch.stack(context_x), torch.stack(context_adj)
        print(f"Context set: {context_x.shape}, {context_adj.shape}")

        context_dataset = TensorDataset(context_x, context_adj)

        print(f"Caching contextset to {contextset_cache_file}")
        with open(contextset_cache_file, "wb") as f:
            pickle.dump(context_dataset, f)

    if get_graph_list:
        context_dataloader = [
            nx.from_numpy_matrix(np.array(adj)) for x, adj in context_dataset
        ]

    else:

        # push datasets to device
        print("Moving data to GPU")
        context_dataset.tensors = tuple(
            tensor.to(f"cuda:{device[0]}") for tensor in context_dataset.tensors
        )

        for tensor in context_dataset.tensors:
            print(tensor.shape, tensor.dtype, tensor.device)

        context_dataloader = DataLoader(
            context_dataset, batch_size=config.data.context_size, shuffle=True
        )

    print(f"{time() - start_time:.2f} sec elapsed for data loading")

    return context_dataloader
