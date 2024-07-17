import torch
from rdkit import Chem
from rdkit.Chem import Draw

from data import xyz2mol
from data.aromatic_dataloader import (
    create_data_loaders,
    RINGS_LIST,
    ATOMS_LIST,
)
from data.ring import RINGS_DICT
from utils.args_edm import Args_EDM
from utils.ring_graph import NO_ORIENTATION_RINGS
from utils.helpers import positions2adj
import matplotlib.pyplot as plt
import numpy as np

hexagon = np.array(
    [
        [6.92302547e-01, -1.19910074e00],
        [-6.92299212e-01, -1.19910016e00],
        [-1.38459997e00, -9.17922477e-07],
        [-6.92301879e-01, 1.19910117e00],
        [6.92298556e-01, 1.19910064e00],
        [1.3846, 0],
    ]
)
pentagon = np.array(
    [[0.3, -1.229], [-0.943, -0.743], [-0.943, 0.742], [0.3, 1.229], [1.286, 0]]
)
square = np.array(
    [
        [5.55111512e-17, 9.47523087e-01],
        [-9.47523087e-01, 5.55111512e-17],
        [-5.55111512e-17, -9.47523087e-01],
        [9.47523087e-01, -5.55111512e-17],
    ]
)
rings = {
    "Bn": hexagon,
    "Bz": hexagon,
    "Pd": hexagon,
    "Pz": hexagon,
    "Db": hexagon,
    "DhDb": hexagon,
    "Th": pentagon,
    "Fu": pentagon,
    "Bl": pentagon,
    "Pl": pentagon,
    "Cbd": square,
}


def align_to_xy_plane(x):
    """
    Rotate the molecule into xy-plane.

    """
    I = np.zeros((3, 3))  # set up inertia tensor I
    com = np.zeros(3)  # set up center of mass com

    # calculate moment of inertia tensor I
    for i in range(x.shape[0]):
        atom = x[i]
        I += np.array(
            [
                [(atom[1] ** 2 + atom[2] ** 2), -atom[0] * atom[1], -atom[0] * atom[2]],
                [-atom[0] * atom[1], (atom[0] ** 2 + atom[2] ** 2), -atom[1] * atom[2]],
                [-atom[0] * atom[2], -atom[1] * atom[2], atom[0] ** 2 + atom[1] ** 2],
            ]
        )

        com += atom
    com = com / len(com)

    # extract eigenvalues and eigenvectors for I
    # np.linalg.eigh(I)[0] are eigenValues, [1] are eigenVectors
    eigenVectors = np.linalg.eigh(I)[1]
    eigenVectorsTransposed = np.transpose(eigenVectors)

    a = []
    for i in range(x.shape[0]):
        xyz = x[i]
        a.append(np.dot(eigenVectorsTransposed, xyz - com))
    return np.stack(a)


def plot_goa(x, bonds=[], atom_types=None):
    plt.scatter(x[:, 0], x[:, 1])
    for b in bonds:
        plt.plot(x[b, 0], x[b, 1], c="b")
    if atom_types is not None:
        for i, a in enumerate(atom_types):
            plt.text(x[i, 0], x[i, 1], a)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def rotation_2d(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )


def lineseg_dists(p, a, b):
    # Handle case where p is a single point, i.e. 1d array.
    p = np.atleast_2d(p)

    if np.all(a == b):
        return np.linalg.norm(p - a, axis=1)

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d[0])
    t = np.dot(p - b, d[0])

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(p))])

    # perpendicular distance component, as before
    # note that for the 3D case these will be vectors
    c = np.cross(p - a, d)

    # use hypot for Pythagoras to improve accuracy
    return np.hypot(h, c)


def gor2goa(x, rings_types, dataset="cata", tol=0.1):
    if dataset == "cata":
        n = x.shape[0]
    else:
        n = x.shape[0] // 2

    _, adj = positions2adj(x[None, :n], rings_types[None, :n], dataset=dataset, tol=tol)
    adj = adj[0]

    x = align_to_xy_plane(x.numpy())[:, :2]
    # plot_goa(x)
    orientation = x[n:]
    x = x[:n]

    atoms = np.zeros([0, 2])
    atoms_types = []
    bonds = []
    rings_atoms_idxs = {}
    for i in range(x.shape[0]):
        ring_type = RINGS_LIST[dataset][rings_types[i]]
        ring = rings[ring_type].copy()
        if ring_type in NO_ORIENTATION_RINGS:
            # neighbor
            if adj.shape[0] == 1:
                angle = 0
            else:
                j = adj[i].nonzero()[0, 0]
                angle = np.arctan2(x[j, 1] - x[i, 1], x[j, 0] - x[i, 0])
            if ring_type == "Bn":
                angle += np.pi / 6
            elif ring_type == "Cbd":
                angle += np.pi / 4
            else:
                raise ValueError
        else:
            hetroatom_coord = orientation[i]
            angle = np.arctan2(
                hetroatom_coord[1] - x[i, 1], hetroatom_coord[0] - x[i, 0]
            )
            # angle -= np.pi / 2

        # rotate ring to the correct orientation
        ring = ring @ rotation_2d(-angle)
        ring += x[i]
        rings_atoms_idxs[i] = list(
            range(atoms.shape[0], atoms.shape[0] + ring.shape[0])
        )
        atoms = np.concatenate([atoms, ring], axis=0)

        s_idx = atoms.shape[0] - ring.shape[0]
        for j in range(ring.shape[0] - 1):
            bonds.append([s_idx + j, s_idx + j + 1])
        bonds.append([s_idx + ring.shape[0] - 1, s_idx])

        atoms_types += RINGS_DICT[ring_type]

        # add H's
        if ring_type in ["Bl", "Pl"]:
            atoms = np.concatenate([atoms, np.zeros([1, 2])], axis=0)
            atoms_types.append("H")
            bonds.append([s_idx + 4, atoms.shape[0] - 1])
        elif ring_type == "DhDb":
            atoms = np.concatenate([atoms, np.zeros([2, 2])], axis=0)
            atoms_types += ["H", "H"]
            bonds.append([s_idx + 2, atoms.shape[0] - 2])
            bonds.append([s_idx + 5, atoms.shape[0] - 1])

    # plot_goa(atoms, bonds, atoms_types)

    # remove duplicates
    adj_u = np.triu(adj)
    if adj.shape[0] == 1:
        ring_bonds = []
    else:
        ring_bonds = list(zip(*adj_u.nonzero()))
    i_idxs, j_idxs = [], []
    for i, j in ring_bonds:
        i_atoms = rings_atoms_idxs[i]
        j_atoms = rings_atoms_idxs[j]
        i_coords = atoms[i_atoms]
        j_coords = atoms[j_atoms]

        # create line between rings centers and find the closest points
        p1, p2 = x[i][None, :], x[j][None, :]
        di = lineseg_dists(i_coords, p1, p2)
        dj = lineseg_dists(j_coords, p1, p2)
        d_i = np.cross(p2 - p1, p1 - i_coords) / np.linalg.norm(p2 - p1)
        d_j = np.cross(p2 - p1, p1 - j_coords) / np.linalg.norm(p2 - p1)
        di2 = di.copy()
        dj2 = dj.copy()
        di[d_i > 0] = np.inf
        dj[d_j > 0] = np.inf
        di2[d_i < 0] = np.inf
        dj2[d_j < 0] = np.inf

        i_idxs += [i_atoms[di.argmin()], i_atoms[di2.argmin()]]
        j_idxs += [j_atoms[dj.argmin()], j_atoms[dj2.argmin()]]

    new_atoms = []
    new_atoms_type = []
    atoms_map = {}
    for i, j in zip(i_idxs, j_idxs):
        new_atoms.append((atoms[i] + atoms[j]) / 2)
        new_atoms_type.append(atoms_types[i])
        atoms_map[i] = len(new_atoms) + len(atoms) - 1
        atoms_map[j] = len(new_atoms) + len(atoms) - 1
        atoms[i] = 0
        atoms[j] = 0

    if len(new_atoms) > 0:
        atoms = np.concatenate([atoms, np.stack(new_atoms, axis=0)], axis=0)
    atoms_types = atoms_types + new_atoms_type
    atoms_types = [ATOMS_LIST[dataset].index(t) for t in atoms_types]
    bonds = [[atoms_map.get(i, i), atoms_map.get(j, j)] for i, j in bonds]

    idx_delete = i_idxs + j_idxs
    atoms = {i: a for i, a in enumerate(atoms) if i not in idx_delete}
    atoms_types = {i: a for i, a in enumerate(atoms_types) if i not in idx_delete}
    idx = list(atoms.keys())
    bonds = [[idx.index(i), idx.index(j)] for i, j in bonds]
    atoms = np.stack(list(atoms.values()), axis=0)
    atoms_types = list(atoms_types.values())

    # remove duplicate bonds
    bonds = [tuple(sorted(a)) for a in bonds]
    bonds = list(set(bonds))
    # plot_goa(atoms, bonds, [ATOMS_LIST[dataset][t] for t in atoms_types])

    return torch.tensor(atoms), torch.tensor(atoms_types), bonds


def smiles2inchi(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchi(m)


def draw_mol(mol, title=""):
    img = Draw.MolToImage(mol)
    plt.imshow(img)
    plt.title(title)
    plt.show()


def build_molecule_aromatic(atom_types, bonds, dataset):
    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(ATOMS_LIST[dataset][atom.item()])
        mol.AddAtom(a)

    for bond in bonds:
        if atom_types[bond[0]] == "H" or atom_types[bond[1]] == "H":
            mol.AddBond(bond[0], bond[1], Chem.rdchem.BondType.SINGLE)
        else:
            mol.AddBond(bond[0], bond[1], Chem.rdchem.BondType.AROMATIC)

    atoms = list(mol.GetAtoms())
    for atom in atoms:
        if atom.GetDegree() == 2 and atom.GetSymbol() == "C":
            h = Chem.Atom("H")
            h_idx = mol.AddAtom(h)
            mol.AddBond(atom.GetIdx(), h_idx, Chem.rdchem.BondType.SINGLE)

    return mol


def rdkit_valid(atoms_types, bonds, dataset="cata"):
    valid = []
    for a_types, b in zip(atoms_types, bonds):
        mol = build_molecule_aromatic(a_types, b, dataset)
        AC = Chem.GetAdjacencyMatrix(mol)
        atoms = [xyz2mol.int_atom(atom.GetSymbol()) for atom in mol.GetAtoms()]
        rwmol = Chem.RWMol()
        for atom in mol.GetAtoms():
            a = Chem.Atom(atom.GetSymbol())
            rwmol.AddAtom(a)

        is_valid = False
        try:
            mol = xyz2mol.AC2mol(rwmol, AC, atoms, 0)
            if len(mol) == 1:
                mol = mol[0]
                Chem.SanitizeMol(mol)
                if len(Chem.GetMolFrags(mol, asMols=True)) == 1:
                    is_valid = True
        except:
            pass

        if is_valid:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            valid.append(smiles2inchi(smiles))

    return valid, len(valid) / len(atoms_types)


if __name__ == "__main__":
    args = Args_EDM().parse_args()
    args.device = "cpu"
    args.tol = 0.1
    args.sample_rate = 0.01
    # args.batch_size = 2
    # args.num_workers = 0
    # args.dataset = "cata"
    # args.orientation = False
    # args.target_features = "GAP_eV"
    train_loader, val_loader, test_loader = create_data_loaders(args)

    np.random.seed(0)
    for i in np.random.randint(0, len(train_loader.dataset), 10):
        x, node_mask, edge_mask, node_features, y = train_loader.dataset[i]
        node_mask = node_mask.bool()
        atoms_positions, atoms_types, bonds = gor2goa(
            x[node_mask], node_features[node_mask].argmax(1), args.dataset
        )
        # np.save('GOA-with-Hs.npy', [atoms_positions, atoms_types, bonds])
        valid, val_ration = rdkit_valid([atoms_types], [bonds], args.dataset)
        print(val_ration)
        # print(valid)

    molecule_list = []
    loader = test_loader
    for x, node_mask, edge_mask, h, y in loader:
        node_mask = node_mask.bool()
        molecule_list += [
            (x[i, node_mask[i]], h[i, node_mask[i]].argmax(1))
            for i in range(x.shape[0])
        ]

    stability_dict, molecule_stable_list = analyze_rdkit_valid_for_molecules(
        molecule_list, args.tol, args.dataset
    )
    for key, value in stability_dict.items():
        try:
            print(f"   {key}: {value:.2%}")
        except:
            pass
