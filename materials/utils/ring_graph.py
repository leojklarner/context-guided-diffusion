from typing import Sequence

import networkx as nx
from torch import zeros, Tensor

from data.mol import Atom
from data.ring import Ring, RINGS_DICT

NO_ORIENTATION_RINGS = ["Bn", "Cbd"]


def get_ring_type(str):
    for key, value in RINGS_DICT.items():
        if sorted(str) == sorted(value):
            return key
    raise NotImplementedError


def get_ringgraph(_atoms: Sequence[Atom], _molgraph: nx.graph) -> nx.graph:
    """
    Generate a graph of rings (Knot Objects).

    in:
    _atoms: A list of Atoms with their xyz coordinates in Angstroms.
    _molgraph: Molecular graph.

    out:
    graph: generated graph of the knots.

    """
    knots = get_rings(_atoms, _molgraph)
    edges = get_rings_connectivity(knots)
    graph = nx.Graph(edges)  # generate mathematical graph as networkx Graph object

    return graph


def get_rings(_atoms: Sequence[Atom], _molgraph: nx.graph) -> Sequence[Ring]:
    """
    Function that gets the geometric center of each ring of the molecule and initializes the Knot Objects for each monocycle.

    in:
    _atoms: A list of Atoms with their xyz coordinates in Angstroms.
    _cycles: A list of monocycles. Each monocycle is a list of atom indices.

    out:
    knots: A list of Knots (= monocycles).

    """
    cycles = nx.minimum_cycle_basis(_molgraph)
    knots = []  # initialize list to return
    i = 0
    for cycle in cycles:
        cycle_atoms = ""
        x_knot = y_knot = z_knot = 0
        for atom in cycle:
            cycle_atoms += _atoms[atom].element
            x_knot += _atoms[atom].x
            y_knot += _atoms[atom].y
            z_knot += _atoms[atom].z
        x = x_knot / len(cycle)
        y = y_knot / len(cycle)
        z = z_knot / len(cycle)

        knot_type = get_ring_type(cycle_atoms)
        if "Db" in knot_type:
            b_ind = cycle_atoms.index("B")
            b_atom = cycle[b_ind]
            b_neighbors = [_atoms[n].element for n in nx.neighbors(_molgraph, b_atom)]
            if "H" in b_neighbors:
                knot_type = "DhDb"
            else:
                knot_type = "Db"

        # save the ring orientation - defined as the on of the hetro atoms coordinates
        if knot_type in NO_ORIENTATION_RINGS:
            # for these rings, the orientation is the centeroid of the ring
            orientation = [[x, y, z]]
        else:
            orientation = [
                _atoms[atom].get_coord()
                for atom in cycle
                if _atoms[atom].element != "C"
            ]

        if len(orientation) == 0:
            raise ValueError("No orientation for ring")

        _knot = Ring(i, knot_type, x, y, z, [_atoms[x] for x in cycle], orientation)
        i += 1
        knots.append(_knot)

    return knots


def get_rings_connectivity(_knots: Sequence[Ring]) -> Sequence[tuple]:
    """
    get_connectivity(_knots: list(Knot)) -> list(tuple)

    Find out which Knot objects are connected and return those connections as a list of tuples ( = edges).

    in:
    _knots: A list of Knot objects.

    out:
    edges: A list of tuples ( = edges) which represent which Knot objects are connected.

    """
    edges = []
    for i in range(len(_knots)):
        for j in range(i + 1, len(_knots)):
            i_atoms = set(_knots[i].atoms)
            j_atoms = set(_knots[j].atoms)
            if i_atoms & j_atoms:
                edges.append((i, j))

    return edges


def get_rings_adj(_knots: Sequence[Ring]) -> Tensor:
    adj = zeros(len(_knots), len(_knots))
    for i in range(len(_knots)):
        for j in range(i + 1, len(_knots)):
            i_atoms = set(_knots[i].atoms)
            j_atoms = set(_knots[j].atoms)
            if i_atoms & j_atoms:
                adj[i, j] = adj[j, i] = 1
    return adj
