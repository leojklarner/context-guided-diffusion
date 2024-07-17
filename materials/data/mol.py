import math
import numpy as np

from dataclasses import dataclass

import torch

__ATOM_LIST__ = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V ",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
]

from rdkit import Chem


class Mol:
    def __init__(self, _atoms):
        if isinstance(_atoms[0], Atom):
            self.atoms = _atoms
        else:
            self.atoms = []
            i = 0
            for atom in _atoms:
                self.atoms.append(Atom(i, atom[0], atom[1], atom[2], atom[3]))
                i += 1

    def __str__(self):
        length = math.floor(math.log10(len(self.atoms))) + 1
        return "\n".join(
            f"{atom.index:{length}}  {atom.element:2} {atom.x:8.5f} {atom.y:8.5f} {atom.z:8.5f}"
            for atom in self.atoms
        )

    def __getitem__(self, index):
        return self.atoms[index]

    def align_to_xy_plane(self):
        """
        Rotate the molecule into xy-plane. In-place operation.

        """

        I = np.zeros((3, 3))  # set up inertia tensor I
        com = np.zeros(3)  # set up center of mass com

        # calculate moment of inertia tensor I
        for atom in self.atoms:
            I += np.array(
                [
                    [(atom.y**2 + atom.z**2), -atom.x * atom.y, -atom.x * atom.z],
                    [-atom.x * atom.y, (atom.x**2 + atom.z**2), -atom.y * atom.z],
                    [-atom.x * atom.z, -atom.y * atom.z, atom.x**2 + atom.y**2],
                ]
            )

            com += [atom.x, atom.y, atom.z]
        com = com / len(com)

        # extract eigenvalues and eigenvectors for I
        # np.linalg.eigh(I)[0] are eigenValues, [1] are eigenVectors
        eigenVectors = np.linalg.eigh(I)[1]
        eigenVectorsTransposed = np.transpose(eigenVectors)

        # transform xyz to new coordinate system.
        # transform v1 -> ex, v2 -> ey, v3 -> ez
        for atom in self.atoms:
            xyz = [atom.x, atom.y, atom.z]
            atom.x, atom.y, atom.z = np.dot(eigenVectorsTransposed, xyz - com)

        z = torch.tensor([atom.z for atom in self.atoms])
        # if z.mean() > 1e-1 or z.std() > 1e-1:
        #     print(z.std())
        #     raise Exception

    def get_coord(self):
        return torch.stack(
            [torch.tensor([atom.x, atom.y, atom.z]) for atom in self.atoms]
        )


@dataclass
class Atom:
    index: int
    element: str
    x: float
    y: float
    z: float

    def __repr__(self):
        return (
            f"{self.index:4} {self.element:2} {self.x:8.5f} {self.y:8.5f} {self.z:8.5f}"
        )

    def __hash__(self):
        return hash(f"{self.index}{self.element}{self.x}{self.y}{self.z}")

    def __eq__(self, other):
        return (
            self.index == other.index
            and self.element == other.element
            and math.isclose(self.x, other.x, 1e-9, 1e-9)
            and math.isclose(self.y, other.y, 1e-9, 1e-9)
            and math.isclose(self.z, other.z, 1e-9, 1e-9)
        )

    def get_coord(self):
        return [self.x, self.y, self.z]


def load_xyz(_path: str) -> Mol:
    """
    load_xyz(_path: str) -> molrepr: Mol

    Load molecule from an XYZ input file and initialize a Mol Object for it.

    in:
    _path: String with the path of the input file.

    out:
    molrepr: Mol Object containing the molecular information (coordinates and elements).

    """

    molrepr = []
    with open(_path, "r") as file:
        for line_number, line in enumerate(file):
            if line_number > 1:
                atomic_symbol, x, y, z = line.split()
                if not atomic_symbol.isalpha():
                    atomic_symbol = int(atomic_symbol)
                    atomic_symbol = str_atom(atomic_symbol)
                molrepr.append(
                    [atomic_symbol.capitalize(), float(x), float(y), float(z)]
                )

    molrepr = Mol(molrepr)
    return molrepr


def from_rdkit(file_path):
    mol = Chem.MolFromMolFile(file_path)
    molrepr = []
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        coords = conf.GetAtomPosition(idx)
        molrepr.append(
            [
                atom.GetSymbol().capitalize(),
                float(coords.x),
                float(coords.y),
                float(coords.z),
            ]
        )
    molrepr = Mol(molrepr)
    atom_connectivity = Chem.GetAdjacencyMatrix(mol)
    return molrepr, atom_connectivity


def str_atom(_atom: int) -> str:
    """
    str_atom(_atom: int) -> atom: str

    Convert integer atom to string atom.

    in:
    _atom: Integer with the atomic number.

    out:
    atom: String with the atomic element.

    """

    atom = __ATOM_LIST__[_atom - 1]
    return atom
