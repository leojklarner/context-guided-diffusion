from typing import Tuple
import json

import torch
from torch import Tensor

from cond_prediction.prediction_args import PredictionArgs
from data.aromatic_dataloader import RINGS_LIST
from utils.args_edm import Args_EDM

bn_bn_dist = {"min": 2.399, "mean": 2.445, "max": 2.481, "thr": 0.01}
bn_bn_angels3_dict = {  # 0.001 and 0.999 quantiles
    "120": (105.772, 133.193),
    "180": (177.333, 183.089),
    "240": (227.120, 255.250),
}
angels3_dict_hetro = {
    "Bl": {
        "140": (127.3096694946289, 145.93600463867188),
    },
    "Bn": {
        "120": (108.33101654052734, 127.21441650390625),
        "180": (170.7755126953125, 180.0),
    },
    "Db": {
        "180": (156.42091369628906, 180.0),
    },
    "Fu": {
        "140": (135.90780639648438, 153.3458251953125),
    },
    "Pl": {
        "140": (134.00990295410156, 151.88079833984375),
    },
    "Bz": {
        "120": (108.01634216308594, 123.69662475585938),
        "180": (169.33651733398438, 179.944580078125),
    },
    "Pz": {
        "180": (168.29324340820312, 180.0),
    },
    "Pd": {
        "120": (108.94857788085938, 126.54322052001953),
        "180": (168.7400360107422, 179.96141052246094),
    },
    "Th": {
        "140": (126.71401977539062, 142.5613555908203),
    },
    "Cbd": {
        "180": (155.19215393066406, 180.0),
    },
}
angels3_dict = {"cata": {"Bn": bn_bn_angels3_dict}, "hetro": angels3_dict_hetro}

angels4_dict = {
    "cata": {  # 0.01 quantile
        "0": 43.943,
        "180": 135.031,
    },
    "hetro": {  # 0.01 quantile
        "0": 42.01443862915039,
        "180": 139.9242706298828,
    },
}
analyzed_rings = {
    "cata": {
        "n_nodes": {
            11: 20559,
            10: 5164,
            9: 1349,
            8: 363,
            7: 108,
            5: 11,
            6: 32,
            3: 2,
            4: 3,
            1: 1,
            2: 1,
        },
        "ring_types": None,  # {0: 303523},
        "distances": None,
        # [270, 434, 972, 1518, 1990, 2758, 3414, 3458, 4022, 4540, 5246, 5532, 538744, 6196, 6542, 6556, 10008, 12276, 11214, 11858, 20486, 416236, 19804, 29788, 188832, 103086, 12392, 12434, 12766, 11822, 8860, 30886, 329638, 68412, 24956, 18418, 76588, 76760, 10590, 10952, 8780, 12796, 102274, 56874, 117212, 48644, 11988, 12820, 25174, 38428, 10530, 8724, 13620, 70440, 32986, 24980, 50426, 18560, 7818, 5448, 7150, 15068, 9172, 12822, 25654, 28590, 11984, 7548, 16716, 8068, 3424, 2026, 2328, 4546, 9494, 13400, 7634, 7602, 2894, 1644, 3890, 3128, 986, 672, 982, 3612, 5154, 2070, 1908, 1324, 456, 182, 656, 850, 248, 212, 1116, 520, 884, 1298]
    },
    "hetro": {
        "n_nodes": {
            10: 56617,
            9: 111471,
            8: 107610,
            7: 66431,
            5: 8622,
            6: 28604,
            4: 1829,
            3: 329,
            2: 51,
        },
    },
}

ring_distances_hetro = {
    "Pl-Bn": (2.13, 2.18),
    "Th-Bn": (2.22, 2.28),
    "Bn-Bn": (2.42, 2.48),
    "Fu-Bn": (2.12, 2.17),
    "Fu-Cbd": (1.61, 1.70),
    "Cbd-Bn": (1.87, 1.95),
    "Bn-Bl": (2.18, 2.26),
    "Pd-Bn": (2.33, 2.39),
    "Db-Bn": (2.51, 2.63),
    "Pz-Bn": (2.38, 2.46),
    "Pz-Db": (2.48, 2.61),
    "Bz-Bn": (2.42, 2.55),
    "Th-Bz": (2.22, 2.34),
    "Db-Bl": (2.27, 2.40),
    "Pl-Cbd": (1.62, 1.70),
    "Db-Cbd": (1.93, 2.09),
    "Th-Bl": (1.99, 2.06),
    "Fu-Db": (2.22, 2.32),
    "Db-Bz": (2.53, 2.69),
    "Th-Fu": (1.93, 1.99),
    "Pd-Bl": (2.10, 2.17),
    "Pz-Pd": (2.29, 2.37),
    "Pz-Bz": (2.38, 2.53),
    "Bl-Bl": (1.96, 2.06),
    "Db-Db": (2.53, 2.78),
    "Th-Db": (2.31, 2.43),
    "Cbd-Bz": (1.83, 2.00),
    "Bz-Bl": (2.20, 2.32),
    "Fu-Bz": (2.11, 2.24),
    "Fu-Fu": (1.86, 1.89),
    "Pd-Db": (2.45, 2.54),
    "Th-Pd": (2.13, 2.18),
    "Pz-Bl": (2.14, 2.24),
    "Pz-Fu": (2.08, 2.15),
    "Pz-Pl": (2.09, 2.17),
    "Pd-Bz": (2.33, 2.46),
    "Th-Cbd": (1.70, 1.78),
    "Th-Pz": (2.17, 2.26),
    "Pl-Pd": (2.05, 2.09),
    "Th-Pl": (1.95, 1.99),
    "Bz-Bz": (2.47, 2.61),
    "Pz-Pz": (2.33, 2.42),
    "Pd-Fu": (2.03, 2.08),
    "Fu-Bl": (1.89, 1.97),
    "Pl-Fu": (1.87, 1.90),
    "Pl-Bl": (1.91, 1.98),
    "Pl-Db": (2.22, 2.34),
    "Th-Th": (2.03, 2.08),
    "Cbd-Cbd": (1.25, 1.46),
    "Pl-Bz": (2.13, 2.25),
    "Pd-Cbd": (1.80, 1.84),
    "Pz-Cbd": (1.84, 1.93),
    "Pl-Pl": (1.89, 1.91),
    "Pd-Pd": (2.25, 2.35),
    "Cbd-Bl": (1.65, 1.75),
}
ring_distances_cata = {
    "Bn-Bn": (2.42, 2.48),
}
ring_distances = {
    "cata": ring_distances_cata,
    "peri": ring_distances_cata,
    "hetro": ring_distances_hetro,
}


def coord2distances(x):
    x = x.unsqueeze(2)
    x_t = x.transpose(1, 2)
    dist = (x - x_t) ** 2
    dist = torch.sqrt(torch.sum(dist, 3))
    return dist


def positions2adj(
    x: Tensor, ring_type, tol=0.1, dataset="cata"
) -> Tuple[Tensor, Tensor]:
    dist = coord2distances(x)
    if len(ring_type.shape) == 3:
        ring_type = ring_type.argmax(2)
    adj = torch.zeros(dist.shape[0], dist.shape[1], dist.shape[1])
    for b in range(dist.shape[0]):
        for i in range(dist.shape[1]):
            for j in range(i + 1, dist.shape[1]):
                si = RINGS_LIST[dataset][ring_type[b, i]]
                sj = RINGS_LIST[dataset][ring_type[b, j]]
                key = f"{si}-{sj}"
                if key not in ring_distances[dataset]:
                    key = f"{sj}-{si}"
                if key in ring_distances[dataset] and ring_distances[dataset][key][
                    0
                ] * (1 - tol) < dist[b, i, j] < ring_distances[dataset][key][1] * (
                    1 + tol
                ):
                    adj[b, i, j] = 1
                    adj[b, j, i] = 1

    return dist, adj


def switch_grad_off(models):
    for m in models:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False


def get_edm_args(exp_dir_path):
    args = Args_EDM().parse_args([])
    with open(exp_dir_path + "/args.txt", "r") as f:
        args.__dict__ = json.load(f)
    args.restore = True
    args.exp_dir = exp_dir_path
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    return args


def get_cond_predictor_args(exp_dir_path):
    args = PredictionArgs().parse_args([])
    with open(exp_dir_path + "/args.txt", "r") as f:
        args.__dict__ = json.load(f)
    args.restore = True
    args.exp_dir = exp_dir_path
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    return args
