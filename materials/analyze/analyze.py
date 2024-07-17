from argparse import Namespace

from torch import Tensor
from tqdm import tqdm

import torch
import networkx as nx
import numpy as np

from data.aromatic_dataloader import create_data_loaders, RINGS_LIST, get_splits
from data.gor2goa import rdkit_valid, gor2goa, smiles2inchi
from utils.args_edm import Args_EDM
from utils.helpers import (
    positions2adj,
    ring_distances,
    angels3_dict,
    angels4_dict,
)


def check_angels3(angels3: Tensor, tol=0.1, dataset="cata") -> bool:
    a3_dict = angels3_dict[dataset]
    if len(angels3) == 0:
        return True
    symbols = [a[0] for a in angels3]
    for symbol in set(symbols):
        a3_symbol = torch.stack([a[1] for a in angels3 if a[0] == symbol])
        conds = [
            torch.logical_and(
                q_low * (1 - tol) <= a3_symbol, a3_symbol <= q_high * (1 + tol)
            )
            for q_low, q_high in a3_dict[symbol].values()
        ]
        if not torch.stack(conds).any(dim=0).all():
            return False
    return True


def check_angels4(angels4: Tensor, tol=0.1, dataset="cata") -> bool:
    if len(angels4) == 0 or dataset == "hetro":
        return True
    a4_dict = angels4_dict[dataset]
    angels4 = torch.stack([a for s, a in angels4])
    cond = torch.logical_or(
        a4_dict["180"] * (1 - tol) <= angels4, angels4 <= a4_dict["0"] * (1 + tol)
    )
    return cond.all()


def check_stability(positions, ring_type, tol=0.1, dataset="cata") -> dict:
    results = {
        "orientation_nodes": True,
        "dist_stable": False,
        "connected": False,
        "angels3": False,
        "angels4": False,
    }
    if isinstance(positions, np.ndarray):
        positions = Tensor(positions)
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    if len(ring_type.shape) == 2:
        ring_type = ring_type.argmax(1)

    if dataset != "cata":  # orientation nodes
        n_rings = torch.div(positions.shape[0], 2, rounding_mode="trunc")
        positions = positions[:n_rings]
        # check orientation nodes
        orientation_ring_type = len(RINGS_LIST["hetro"]) - 1
        if (
            set(ring_type[n_rings:].numpy()) != set([orientation_ring_type])
            or orientation_ring_type in ring_type[:n_rings]
        ):
            results["orientation_nodes"] = False
            return results
        ring_type = ring_type[:n_rings]
    n_rings = positions.shape[0]
    dist, adj = positions2adj(
        positions[None, :, :], ring_type[None, :], tol, dataset=dataset
    )
    dist = dist[0]
    adj = adj[0]
    min_dist = min([r[0] for r in ring_distances[dataset].values()])
    if ((dist < min_dist * (1 - tol)) * (1 - torch.eye(n_rings))).bool().any():
        return results
    else:
        results["dist_stable"] = True

    g = nx.from_numpy_array(adj.numpy())
    if not nx.is_connected(g):
        return results
    else:
        results["connected"] = True

    angels3, angels4 = get_angels(
        positions[None, :, :], ring_type[None, :], adj[None, :, :], dataset=dataset
    )
    results["angels3"] = check_angels3(angels3, tol, dataset=dataset)
    results["angels4"] = check_angels4(angels4, tol, dataset=dataset)
    return results


def main_check_stability(args, tol=0.1, rdkit=False):
    train_dataloader, val_dataloader, test_dataloader = create_data_loaders(args)

    def test_validity_for(dataloader):
        molecule_list = []
        for i, (x, node_mask, edge_mask, node_features, y) in enumerate(dataloader):
            for j in range(x.size(0)):
                positions = x[j].view(-1, 3)
                one_hot = node_features[j].type(torch.float32)
                mask = node_mask[j].flatten().bool()
                positions, one_hot = positions[mask], one_hot[mask]
                atom_type = torch.argmax(one_hot, dim=1).numpy()
                molecule_list.append((positions, atom_type))
        validity_dict, molecule_stable_list = analyze_validity_for_molecules(
            molecule_list, tol=tol, dataset=args.dataset
        )
        del validity_dict["molecule_stable_bool"]
        print("Validity:")
        print(validity_dict)

        if rdkit:
            validity_dict, molecule_stable_list = analyze_rdkit_validity_for_molecules(
                molecule_list, tol=tol, dataset=args.dataset
            )
            del validity_dict["molecule_stable_bool"]
            print("Validity rdkit:")
            print(validity_dict)

    print("\nFor train")
    test_validity_for(train_dataloader)
    print("\nFor val")
    test_validity_for(val_dataloader)
    print("\nFor test")
    test_validity_for(test_dataloader)


def analyze_validity_for_molecules(molecule_list, tol=0.1, dataset="cata"):
    n_samples = len(molecule_list)
    molecule_stable_list = []
    molecule_stable_bool = []

    n_molecule_stable = (
        n_dist_stable
    ) = n_connected = n_angel3 = n_angel4 = n_orientation = 0

    # with tqdm(molecule_list, unit="mol") as tq:
    for i, (x, atom_type) in enumerate(molecule_list):
        validity_results = check_stability(x, atom_type, tol=tol, dataset=dataset)

        molecule_stable = all(validity_results.values())
        n_molecule_stable += int(molecule_stable)
        n_dist_stable += int(validity_results["dist_stable"])
        n_connected += int(validity_results["connected"])
        n_angel3 += int(validity_results["angels3"])
        n_angel4 += int(validity_results["angels4"])
        n_orientation += int(validity_results["orientation_nodes"])

        molecule_stable_bool.append(molecule_stable)
        if molecule_stable:
            molecule_stable_list.append((x, atom_type))

    # Validity
    validity_dict = {
        "mol_stable": n_molecule_stable / float(n_samples),
        "orientation_nodes": n_orientation / float(n_samples),
        "dist_stable": n_dist_stable / float(n_samples),
        "connected": n_connected / float(n_samples),
        "angels3": n_angel3 / float(n_samples),
        "angels4": n_angel4 / float(n_samples),
        "molecule_stable_bool": molecule_stable_bool,
    }

    # print('Validity:', validity_dict)

    return validity_dict, molecule_stable_list


def analyze_rdkit_validity_for_molecules(
    molecule_list, tol=0.1, dataset="cata", calc_novelty=False
):
    n_samples = len(molecule_list)
    molecule_valid_list = []
    molecule_valid_bool = []
    valid_inchi = []

    with tqdm(molecule_list, unit="mol") as tq:
        for i, (x, rings_type) in enumerate(tq):
            try:
                atoms, atoms_types, bonds = gor2goa(
                    x,
                    rings_type,
                    tol=tol,
                    dataset=dataset,
                )
                valid, val_ration = rdkit_valid([atoms_types], [bonds], dataset)
                molecule_valid = len(valid) > 0
            except:
                molecule_valid = False

            molecule_valid_bool.append(molecule_valid)
            if molecule_valid:
                molecule_valid_list.append((x, rings_type))
                valid_inchi += valid

    unique = set(valid_inchi)

    validity_dict = {
        "mol_valid": len(valid_inchi) / float(n_samples),
        "mol_unique": len(unique) / max(len(valid_inchi), 1),
        "molecule_valid_bool": molecule_valid_bool,
        "valid_inchi": valid_inchi,
    }
    if calc_novelty:
        df_train = get_splits(
            Namespace(dataset=dataset, target_features=None, max_nodes=11)
        )[0]
        if "inchi" in df_train.columns:
            train_inchi = df_train["inchi"].tolist()
        else:
            train_smiles = df_train["smiles"].tolist()
            train_inchi = [smiles2inchi(s) for s in train_smiles]

        novel = set(valid_inchi) - set(train_inchi)
        validity_dict["mol_novel"] = len(novel) / max(len(valid_inchi), 1)

    return validity_dict, molecule_valid_list


def angel3(p):
    v1 = p[0] - p[1]
    v2 = p[2] - p[1]
    dot_product = torch.dot(v1, v2)
    norm_product = torch.norm(v1) * torch.norm(v2)
    a = torch.rad2deg(torch.acos(dot_product / norm_product))
    return a if a >= 0 else a + 360


def angel4(p):
    """Praxeolitic formula for dihedral angle"""
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= torch.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - torch.dot(b0, b1) * b1
    w = b2 - torch.dot(b2, b1) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.dot(v, w)
    y = torch.dot(torch.cross(b1, v), w)
    return torch.rad2deg(torch.atan2(y, x)).abs()


def find_triplets_quads(adj: Tensor, x: Tensor, ring_types: Tensor, dataset="cata"):
    rings_list = RINGS_LIST[dataset]
    if len(ring_types.shape) == 2:
        ring_types = ring_types.argmax(1)
    rings = [rings_list[i] for i in ring_types]
    g = nx.from_numpy_array(adj.numpy())
    triplets = []
    for n1, n2 in nx.bfs_edges(g, 0):
        for n3 in g.neighbors(n1):
            if n3 != n2:
                triplets.append((n2, n1, n3))
        for n3 in g.neighbors(n2):
            if n3 != n1:
                triplets.append((n1, n2, n3))
    triplets = [(n1, n2, n3) if n1 < n3 else (n3, n2, n1) for n1, n2, n3 in triplets]
    triplets = list(set(triplets))
    # save all the angels with the center ring type
    # if any([angel3(x[nodes, :])<60 for nodes in triplets]):
    #     print([angel3(x[nodes, :]) for nodes in triplets])
    angels3 = [(rings[nodes[1]], angel3(x[nodes, :])) for nodes in triplets]

    angular_triplets = [t for t in triplets if not 170 < angel3(x[t, :]) < 190]
    # quads
    quads = []
    for n1, n2, n3 in angular_triplets:
        for n4 in g.neighbors(n1):
            if n4 not in [n2, n3]:
                # check the new angle is not linear
                if not 175 < angel3(x[[n4, n1, n2]]) < 185:
                    quads.append((n4, n1, n2, n3))
        for n4 in g.neighbors(n3):
            if n4 not in [n1, n2]:
                # check the new angle is not linear
                if not 175 < angel3(x[[n2, n3, n4]]) < 185:
                    quads.append((n1, n2, n3, n4))
    quads = [
        (n1, n2, n3, n4) if n1 < n4 else (n4, n3, n2, n1) for n1, n2, n3, n4 in quads
    ]
    quads = list(set(quads))
    angels4 = [
        ([rings[nodes[i]] for i in range(4)], angel4(x[nodes, :])) for nodes in quads
    ]

    # if any([80<a[1]<100 for a in angels4]):
    #     print([a[1] for a in angels4])

    return angels3, angels4


def get_angels(xs: Tensor, ring_types, adjs, node_masks=None, dataset="cata"):
    """Extract list of angels from batch of nodes"""
    # _, adjs = positions2adj(xs, ring_types, dataset)
    angels3 = []
    angels4 = []
    for i in range(xs.shape[0]):
        adj = adjs[i]
        x = xs[i]
        ring_type = ring_types[i]
        if node_masks is not None:
            node_mask = node_masks[i].bool()
            adj = adj[node_mask][:, node_mask]
            x = x[node_mask]
            ring_type = ring_type[node_mask]
        a3, a4 = find_triplets_quads(adj, x, ring_type, dataset)
        angels3 += a3
        angels4 += a4

    return angels3, angels4


if __name__ == "__main__":
    args = Args_EDM().parse_args()
    args.dataset = "cata"
    args.target_feature = "GAP_eV"
    main_check_stability(args)
