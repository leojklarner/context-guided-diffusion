from typing import List, Tuple, Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R
import networkx as nx

RINGS_DICT = {
    "Bn": "CCCCCC",  # benzene
    "Pl": "CCCCN",  # pyrrole
    "Bl": "CCCCB",  # borole
    "Fu": "CCCCO",  # furan
    "Th": "CCCCS",  # thiophene
    "Pd": "CCCCCN",  # pyrazine
    "Pz": "CCNCCN",  # pyridine
    "Bz": "CCCCCB",  # borabenzene
    "DhDb": "CCBCCB",  # dHdiborine
    "Db": "CCBCCB",  # diborine
    "Cbd": "CCCC",  # cyclobutadiene
}


class Ring:
    def __init__(
        self,
        index: int,
        cycle_type: str,
        x: float,
        y: float,
        z: float,
        atoms: list = [],
        orientation: list = [],
    ):
        self.index = index
        self.cycle_type = cycle_type
        self.x = x
        self.y = y
        self.z = z
        self.atoms = atoms
        self.orientation = orientation

    def __repr__(self):
        atoms = "\n  ".join([str(atom) for atom in self.atoms])
        return f"Knot(\n  index: {self.index}\n  cycle_type: {self.cycle_type}\n  center: {self.x:8.5f} {self.y:8.5f} {self.z:8.5f}\n  atoms:\n  {atoms}\n  )"

    def __str__(self):
        atoms = "\n  ".join([str(atom) for atom in self.atoms])
        return f"  index: {self.index}\n  cycle_type: {self.cycle_type}\n  center: {self.x:8.5f} {self.y:8.5f} {self.z:8.5f}\n  atoms:\n  {atoms}"

    def get_opposing_edge(self, _edge):
        """
        Returns the opposing edge or atom (for 5-membered ring case B) of the inputed edge.

        """

        opposite_atoms = []

        n_atoms = len(self.atoms)
        n_half = n_atoms // 2

        if n_atoms == 5:
            if self.cycle_type in ["furb", "pylb", "thib"]:
                for atom in self.atoms:
                    if atom.element in ["O", "N", "S"]:
                        return (atom,)

            else:
                for atom in self.atoms:
                    if not (
                        atom.index == _edge[0].index
                        or atom.index == _edge[1].index
                        or atom.element in ["N", "O", "S"]
                    ):
                        opposite_atoms.append(atom)
                return (atom for atom in opposite_atoms)

        else:
            a = self.atoms.index(_edge[0])
            b = self.atoms.index(_edge[1])

            if max(a, b) == n_atoms - 1:
                if min(a, b) == 0:
                    start_index = 0
                else:
                    start_index = max(a, b)
            else:
                start_index = max(a, b)

            opposite_atoms.append((start_index + n_half) % n_atoms)
            opposite_atoms.append((start_index + n_half - 1) % n_atoms)
            return (self.atoms[atom] for atom in opposite_atoms)

    def get_coord(self):
        return [self.x, self.y, self.z]


class RingGraph:
    def __init__(self, knots: Sequence[Ring], edges: Sequence[Tuple[int, int]]):
        self.knots = knots
        self.edges = edges
        self.align_to_xy_plane()
        self.align_first_angle(True)

        self.graph = nx.Graph()
        self.graph.add_nodes_from([(i, {"knot": k}) for (i, k) in enumerate(knots)])
        self.graph.add_edges_from(edges)

    def dfs_edges(self):
        return list(nx.dfs_edges(self.graph, source=0))

    def align_to_xy_plane(self):
        """
        Rotate the graph into xy-plane. In-place operation.

        """

        I = np.zeros((3, 3))  # set up inertia tensor I
        com = np.zeros(3)  # set up center of mass com

        # calculate moment of inertia tensor I
        for knot in self.knots:
            x, y, z = knot.get_coord()
            I += np.array(
                [
                    [(y**2 + z**2), -x * y, -x * z],
                    [-x * y, (x**2 + z**2), -y * z],
                    [-x * z, -y * z, x**2 + y**2],
                ]
            )

            com += [x, y, z]
        com = com / len(com)

        # extract eigenvalues and eigenvectors for I
        # np.linalg.eigh(I)[0] are eigenValues, [1] are eigenVectors
        eigenVectors = np.linalg.eigh(I)[1]
        eigenVectorsTransposed = np.transpose(eigenVectors)

        # transform xyz to new coordinate system.
        # transform v1 -> ex, v2 -> ey, v3 -> ez
        for knot in self.knots:
            xyz = knot.get_coord()
            knot.x, knot.y, knot.z = np.dot(eigenVectorsTransposed, xyz - com)

        # z = self.get_coord()[:, 2]
        # if z.mean() > 1e-1 or z.std() > 1e-1:
        #     # print(z.std())
        #     raise Exception

    def first_angle(self):
        if len(self.knots) == 1:
            return 0
        else:
            return self.get_angle(self.edges[0][0], self.edges[0][1])

    def get_angle(self, i: int, j: int):
        x0, y0, _ = self.knots[i].get_coord()
        x1, y1, _ = self.knots[j].get_coord()
        return np.arctan2(y1 - y0, x1 - x0)

    def align_first_angle(self, random_angle=False):
        """
        Rotate the graph so the first angle is one of [0, pi/3, 2pi/3, pi 4pi/3, 5pi,3].
        In-place operation.
        """
        if len(self.knots) == 1:
            return

        angle = self.first_angle()
        if random_angle:
            desired_angle = np.random.randint(6) * np.pi / 3
        else:
            desired_angle = round(angle / (np.pi / 3)) * np.pi / 3
        rotation_needed = desired_angle - angle
        rotation = R.from_rotvec(rotation_needed * np.array([0, 0, 1]))

        # transform xyz to new coordinate system.
        # transform v1 -> ex, v2 -> ey, v3 -> ez
        for knot in self.knots:
            xyz = knot.get_coord()
            knot.x, knot.y, knot.z = rotation.apply(xyz)

    def get_coord(self):
        return np.stack([k.get_coord() for k in self.knots])
