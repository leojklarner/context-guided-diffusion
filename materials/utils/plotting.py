import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from torch import Tensor

from data.aromatic_dataloader import ATOMS_LIST, RINGS_LIST
from data.gor2goa import gor2goa, rdkit_valid

# from matplotlib import cm
# import networkx as nx

from utils.helpers import positions2adj


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


def plot_grap_of_rings_inner(
    ax,
    x: Tensor,
    atom_type,
    title="",
    tol=0.1,
    axis_lim=10,
    align=True,
    dataset="cata",
    adj=None,
):
    x = torch.clamp(x, min=-1e5, max=1e5)
    rings_list = RINGS_LIST["hetro"]
    orientation = dataset != "cata"
    if orientation:
        n = x.shape[0] // 2
        if adj is None:
            _, adj = positions2adj(
                x[None, :n, :], atom_type[None, :n], tol=tol, dataset=dataset
            )
            adj = adj[0]
            adj = torch.cat(
                [
                    torch.cat([adj, torch.eye(n)], dim=1),
                    torch.cat([torch.eye(n), torch.zeros(n, n)], dim=1),
                ],
                dim=0,
            )
    elif adj is None:
        _, adj = positions2adj(
            x[None, :, :], atom_type[None, :], tol=tol, dataset=dataset
        )
        adj = adj[0]

    x = x.cpu().numpy()
    if align:
        x = align_to_xy_plane(x)
        x -= x.mean(0)

    ax.scatter(x[:, 0], x[:, 1], c="blue")
    ring_types = [rings_list[i] for i in atom_type]
    for i in range(x.shape[0]):
        ax.text(x[i, 0], x[i, 1], ring_types[i], fontsize=20, ha="center", va="center")

    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[0]):
            if adj[i, j] == 1:
                ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], c="black")

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    return ax


def plot_rdkit(
    x,
    ring_type,
    ax=None,
    filename="mol_rdkit",
    showPlot=False,
    title="",
    tol=0.1,
    dataset="cata",
    addInChi=False,
):
    plt.rcParams.update({"font.size": 22})
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 9))
    atoms_positions, atoms_types, bonds = gor2goa(
        x.detach().cpu(), ring_type.detach().cpu(), dataset, tol
    )
    valid, val_ration = rdkit_valid([atoms_types], [bonds], dataset)
    if len(valid) == 0:
        return
    if addInChi:
        title = title + "\n" + valid[0]
    mol = Chem.MolFromInchi(valid[0])
    img = Chem.Draw.MolToImage(mol)
    # Chem.Draw.MolToFile(mol, filename + ".png")
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # save figure
    if filename:
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.0)

    # show figure
    if showPlot:
        plt.show()


def plot_graph_of_rings(
    x,
    atom_type,
    filename="mol",
    showPlot=False,
    title="",
    tol=0.1,
    axis_lim=10,
    dataset="cata",
    adj=None,
):
    # set parameters
    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 9))
    plot_grap_of_rings_inner(
        ax,
        x.detach().cpu(),
        atom_type.detach().cpu(),
        title,
        tol=tol,
        axis_lim=axis_lim,
        dataset=dataset,
        adj=adj,
    )

    # save figure
    if filename:
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.0)

    # show figure
    if showPlot:
        plt.show()
    plt.close()


def plot_graph_of_atoms(
    x, one_hot, adj, filename=None, showPlot=False, title="", tol=0.1, axis_lim=10
):
    # set parameters
    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 9))
    x = x.cpu().numpy()
    x = align_to_xy_plane(x)

    ax.scatter(x[:, 0], x[:, 1], c="blue")
    atom_types = [ATOMS_LIST["hetro"][i] for i in one_hot.argmax(1)]
    for i in range(x.shape[0]):
        ax.text(x[i, 0], x[i, 1], atom_types[i], fontsize=20, ha="center", va="center")

    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[0]):
            if adj[i, j] == 1:
                ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], c="black")

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)

    # save figure
    if filename:
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.0)

    # show figure
    if showPlot:
        plt.show()
    plt.close()


def plot_graph_of_rings_3d(
    x,
    atom_type,
    filename=None,
    showPlot=False,
    title="",
    tol=0.1,
    axis_lim=6,
    dataset="cata",
    colors=False,
):
    rings_list = RINGS_LIST["hetro"]
    orientation = dataset != "cata"
    if orientation:
        n = x.shape[0] // 2
        _, adj = positions2adj(
            x[None, :n, :], atom_type[None, :n], tol=tol, dataset=dataset
        )
        adj = adj[0]
        adj = torch.cat(
            [
                torch.cat([adj, torch.eye(n)], dim=1),
                torch.cat([torch.eye(n), torch.zeros(n, n)], dim=1),
            ],
            dim=0,
        )
    else:
        _, adj = positions2adj(
            x[None, :, :], atom_type[None, :], tol=tol, dataset=dataset
        )
        adj = adj[0]

    x = x.cpu().numpy()

    # set parameters
    plt.rcParams.update({"font.size": 22})

    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(projection="3d")

    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if adj[i, j] == 1:
                plt.plot(
                    [x[i, 0], x[j, 0]],
                    [x[i, 1], x[j, 1]],
                    [x[i, 2], x[j, 2]],
                    c="black",
                )

    ring_types = [rings_list[i] for i in atom_type]
    if colors:
        palette = plt.get_cmap("gist_rainbow")
        palette = plt.cm.colors.ListedColormap(
            [palette(x) for x in np.linspace(0, 1, 12)]
        ).colors
        # palette = [plt.cm.Paired(i) for i in range(12)]
        c = [palette[i] for i in atom_type]
        ax.scatter(
            xs=x[::-1, 0], ys=x[::-1, 1], zs=x[::-1, 2], c=c[::-1], s=400, alpha=0.8
        )
    else:
        ax.scatter(xs=x[:, 0], ys=x[:, 1], zs=x[:, 2], c="blue", s=100)
        for i in range(x.shape[0]):
            ax.text(
                x[i, 0],
                x[i, 1],
                x[i, 2],
                ring_types[i],
                fontsize=20,
                ha="center",
                va="center",
            )

    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if adj[i, j] == 1:
                plt.plot(
                    [x[i, 0], x[j, 0]],
                    [x[i, 1], x[j, 1]],
                    [x[i, 2], x[j, 2]],
                    c="black",
                )

    plt.title(title)
    ax.set_axis_off()
    if axis_lim:
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)

    # save figure
    if filename:
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.0)

    # show figure
    if showPlot:
        plt.show()
    plt.close()


def plot_chain(
    x,
    atom_type,
    dirname,
    filename,
    title="",
    tol=0.1,
    axis_lim=6.0,
    dataset="cata",
    gif=True,
    colors=False,
):
    save_paths = []
    os.makedirs(dirname, exist_ok=True)
    for i in range(x.shape[0]):
        save_paths.append(f"{dirname}/chain{i}.pdf")
        plot_graph_of_rings_3d(
            x[i],
            atom_type[i],
            filename=save_paths[-1],
            tol=tol,
            axis_lim=axis_lim,
            dataset=dataset,
            title=i,
            colors=colors,
        )

    if gif:
        # create gif
        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = f"{dirname}/{filename}.gif"
        print(f"Creating gif with {len(imgs)} images")
        # Add the last frame 10 times so that the final result remains temporally.
        # imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True)

        # delete png files
        for file in save_paths:
            os.remove(file)
