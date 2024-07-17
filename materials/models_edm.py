import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import numpy as np

from edm.egnn.models import EGNN_dynamics
from edm.equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from edm.egnn.egnn_new import EGNN, GNN
from edm.equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
from utils.helpers import analyzed_rings


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        if name == "module":
            return super().__getattr__("module")
        else:
            return getattr(self.module, name)


class DistributionRings:
    def __init__(self, dataset="cata"):
        histogram = analyzed_rings[dataset]["n_nodes"]
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        # entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        # print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


def get_model(args, dataloader_train, in_node_nf, only_norm=False):
    # prop_dist = DistributionProperty(args, dataloader_train, only_norm=only_norm)
    prop_dist = None

    # in_node_nf = dataloader_train.dataset.num_node_features
    # nodes_dist = DistributionRings(getattr(args, "dataset", "cata"))
    nodes_dist = None

    net_dynamics = EGNN_dynamics(
        in_node_nf=in_node_nf,
        n_dims=3,
        device=args.device,
        hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(),
        n_layers=args.n_layers,
        attention=args.attention,
        tanh=args.tanh,
        norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers,
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method,
        coords_range=args.coords_range,
        condition_time=True,
    )

    model = EnVariationalDiffusion(
        dynamics=net_dynamics,
        in_node_nf=in_node_nf,
        n_dims=3,
        timesteps=args.diffusion_steps,
        noise_schedule=args.diffusion_noise_schedule,
        noise_precision=args.diffusion_noise_precision,
        loss_type=args.diffusion_loss_type,
        norm_values=args.normalize_factors,
        include_charges=False,
        device=args.device,
    )

    if args.dp:  # and torch.cuda.device_count() > 1:
        model = MyDataParallel(model)
    if args.restore is not None:
        model_state_dict = torch.load(
            args.exp_dir + "/model.pt", map_location=args.device
        )
        model.load_state_dict(model_state_dict)
        print("EDM Model loaded from", args.exp_dir + "/model.pt")
    return model, nodes_dist, prop_dist


class DistributionProperty:
    def __init__(self, args, dataloader, num_bins=1000, only_norm=False):
        self.num_bins = num_bins
        self.distributions = {}

        xh, node_mask, edge_mask, y = dataloader.dataset.tensors

        self.mean = dataloader.dataset.mean
        self.std = dataloader.dataset.std
        if not only_norm:
            self.args = args
            self.properties = dataloader.dataset.target_features
            for i, prop in enumerate(self.properties):
                self.distributions[prop] = {}
                data = torch.Tensor(dataloader.dataset.df[prop])
                if dataloader.dataset.normalize:
                    data = (data - self.mean[i]) / self.std[i]
                self._create_prob_dist(
                    torch.Tensor(dataloader.dataset.df["n_rings"]),
                    data,
                    self.distributions[prop],
                )

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {"probs": probs, "params": params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins  # min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min) / prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(probs)
        params = [prop_min, prop_max]
        return probs, params

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist["probs"].sample((1,))
            val = self._idx2value(idx, dist["params"], len(dist["probs"].probs))
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def sample_df(self, nodesxsample, split="test"):
        df = getattr(self.args, f"df_{split}")
        vals = []
        for n_nodes in nodesxsample:
            val = df[df.n_rings == n_nodes.item()].sample(1)[self.properties].values
            vals.append(torch.Tensor(val))
        vals = torch.cat(vals, dim=0)
        return self.normalize(vals)

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val

    def unnormalize(self, val):
        val = val * self.std.to(val.device) + self.mean.to(val.device)
        return val

    def normalize(self, val):
        val = (val - self.mean.to(val.device)) / self.std.to(val.device)
        return val


class EmpiricalDistributionProperty:
    def __init__(self, args, dataloader):
        self.distributions = {}
        self.properties = dataloader.dataset.target_features
        self.mean = dataloader.dataset.mean
        self.std = dataloader.dataset.std
        self.args = args
        self.data = torch.Tensor(dataloader.dataset.df[self.properties].values)

    def sample(self):
        return self.sample_batch(1)

    def sample_batch(self, nodesxsample):
        return self.normalize(
            self.data[torch.randperm(self.data.shape[0])[: len(nodesxsample)]]
        )

    def unnormalize(self, val):
        val = val * self.std.to(val.device) + self.mean.to(val.device)
        return val

    def normalize(self, val):
        val = (val - self.mean.to(val.device)) / self.std.to(val.device)
        return val
