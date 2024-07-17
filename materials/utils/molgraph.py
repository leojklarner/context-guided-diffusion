import numpy as np
import networkx as nx

from data.mol import Mol, Atom
from utils.const import __COV_RADII__, __ATOM_LIST__
import torch


def get_molgraph(
    _molrepr: Mol = None, covalency_factor: float = 1.3, skip_hydrogen=False
) -> nx.Graph:
    """
    get_molgraph(_molrepr: Mol, covalency_factor: float) -> mol graph (nx.graph)

    Generate a graph for a molecule.

    in:
    _molrepr: A molecule, represented by a list of Atoms with their xyz coordinates in Angstroms.
    covalency_factor: A bond is identified if the sum of covelent radii times the covalency factor is larger than the distance between atoms.

    out:
    graph: generated graph of the molecule.

    """

    atom_connectivity = get_connectivity_matrix(
        _molrepr.atoms, covalency_factor, skip_hydrogen=skip_hydrogen
    )  # build connectivity matrix
    edges = get_edges(atom_connectivity)  # edges = bonds
    graph = nx.Graph(
        edges
    )  # generate a mathematical Graph representation of the molecule using networkx

    return graph


def get_connectivity_matrix(
    _atoms: Atom, covalency_factor: float = 1.3, skip_hydrogen=False
):
    """
    get_connectivity_matrix(_atoms: list(Atoms), covalency_factor: float, skip_hydrogen: bool = False) -> numpy.ndarray

    Function that loops through the atoms and returns the connectivity matrix. Two atoms are considered bonded when the distance between them is less
    or equal to the sum of their covalent radii multiplied by a covalency factor.

    in:
    _atoms: A list of Atoms with their xyz coordinates in Angstroms.
    covalency_factor: A bond is identified if the sum of covelent radii times the covalency factor is larger than the distance between atoms.
    skip_hydrogen: If True, remove hydrogens completely.

    out:
    connectivity_matrix: A connectivity matrix of dimension len(_atoms) x len(_atoms) where elements are 0 if there is no bond and 1 if there is a bond.
                         Diagonal elements are 0.

    """
    number_of_atoms = len(_atoms)
    connectivity_matrix = np.zeros(
        (number_of_atoms, number_of_atoms), dtype=int
    )  # initialize matrix with 0s

    for i in range(number_of_atoms):
        for j in range(
            i + 1, number_of_atoms
        ):  # start at i+1 because diagonal elements should stay 0
            if skip_hydrogen:  # skip hydrogens if set
                if _atoms[i].element == "H" or _atoms[j].element == "H":
                    continue
            covalency_cutoff = (
                __COV_RADII__[_atoms[i].element] + __COV_RADII__[_atoms[j].element]
            ) * covalency_factor  # determine cutoff for elements i,j
            distance_ij = np.sqrt(
                (_atoms[i].x - _atoms[j].x) ** 2
                + (_atoms[i].y - _atoms[j].y) ** 2
                + (_atoms[i].z - _atoms[j].z) ** 2
            )

            if distance_ij <= covalency_cutoff:
                connectivity_matrix[i, j] = connectivity_matrix[j, i] = 1

    return connectivity_matrix


def get_edges(_atom_connectivity):
    """
    get_edges(_atom_connectivity: numpy.ndarray) -> list(tuple)

    Using the connectivity matrix, this function generates a list of tuple, where every tuple contains the atomic index
    of two atoms bonding.

    in:
    _atom_connectivity: Connectivity matrix.

    out:
    edges: A list of tuples that represent connections in the connectivity matrix, i.e. bonds in the molecule.

    """

    dimension = _atom_connectivity.shape[0]
    edges = []
    for i in range(dimension):
        for j in range(i + 1, dimension):
            if _atom_connectivity[i, j] == 1:
                edges.append((i, j))

    return edges
