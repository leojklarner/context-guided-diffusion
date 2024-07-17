import torch
from sde import VPSDE, VESDE
from models.layers import DenseGCNConv
from utils.graph_utils import mask_x, mask_adjs

"""
Changes to use MVE:

- double output dimensions of the regressor
- move sigmoid from self.final_linear to the forward pass
- change loss computation in regressor_utils.py
- change get_empirical_prior_density in regressor_utils.py, original still there
- change context_logits_mean

"""

class Regressor(torch.nn.Module):
    def __init__(self, max_node_num, max_feat_num, depth, nhid, dropout):
        super().__init__()

        self.linears = torch.nn.ModuleList([torch.nn.Linear(max_feat_num, nhid)])
        for _ in range(depth - 1):
            self.linears.append(torch.nn.Linear(nhid, nhid))

        self.convs = torch.nn.ModuleList(
            [DenseGCNConv(nhid, nhid) for _ in range(depth)]
        )

        dim = max_feat_num + depth * nhid
        dim_out = nhid

        self.sigmoid_linear = torch.nn.Sequential(
            torch.nn.Linear(dim, dim_out), torch.nn.Sigmoid()
        )
        self.tanh_linear = torch.nn.Sequential(
            torch.nn.Linear(dim, dim_out), torch.nn.Tanh()
        )

        self.final_linear = [
            torch.nn.Linear(dim_out, nhid),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(nhid, 2),
        ]
        self.final_linear = torch.nn.Sequential(*self.final_linear)

    def forward(self, x, adj, flags):
        xs = [x]
        out = x
        for lin, conv in zip(self.linears, self.convs):
            out = conv(lin(out), adj)
            out = torch.tanh(out)
            out = mask_x(out, flags)
            xs.append(out)
        out = torch.cat(xs, dim=-1)  # bs, max_feat_num, dim

        sigmoid_out = self.sigmoid_linear(out)
        tanh_out = self.tanh_linear(out)
        out = torch.mul(sigmoid_out, tanh_out).sum(dim=1)
        embeds = torch.tanh(out)

        preds = self.final_linear(embeds)

        # apply sigmoid to mean output as in original paper
        preds[:, 0] = torch.sigmoid(preds[:, 0])

        return preds, embeds


class RegressorEnsemble(torch.nn.Module):
    def __init__(self, max_node_num, max_feat_num, depth, nhid, dropout, ensemble_size):
        super().__init__()
        self.regressors = torch.nn.ModuleList(
            [
                Regressor(max_node_num, max_feat_num, depth, nhid, dropout)
                for _ in range(ensemble_size)
            ]
        )

    def forward(self, x, adj, flags):

        preds = []
        embeds = []
        for regressor in self.regressors:
            pred, embed = regressor(x, adj, flags)
            preds.append(pred)
            embeds.append(embed)

        return preds, embeds


def get_regressor_fn(sde, model):
    model_fn = model

    if isinstance(sde, VPSDE):

        def regressor_fn(x, adj, flags, t):
            pred, embed = model_fn(x, adj, flags)
            return pred, embed

    elif isinstance(sde, VESDE):

        def regressor_fn(x, adj, flags, t):
            pred, embed = model_fn(x, adj, flags)
            return pred, embed

    else:
        raise NotImplementedError(f"SDE class: {sde.__class__.__name__} not supported.")

    return regressor_fn


class RegressorScoreX(torch.nn.Module):
    def __init__(self, sde, Regressor):
        super().__init__()
        self.sde = sde
        self.regressor = get_regressor_fn(sde, Regressor)

    def forward(self, x, adj, flags, t):
        with torch.enable_grad():
            x_para = torch.nn.Parameter(x)
            F, _ = self.regressor(x_para, adj, flags, t)
            # average gradients if model is an ensemble
            if isinstance(F, list):
                F = torch.stack(F, dim=0).mean(dim=0)
            F = F.sum()
            F.backward()
            score = x_para.grad
            score = mask_x(score, flags)
        return score


class RegressorScoreAdj(torch.nn.Module):
    def __init__(self, sde, Regressor):
        super().__init__()
        self.sde = sde
        self.regressor = get_regressor_fn(sde, Regressor)

    def forward(self, x, adj, flags, t):
        with torch.enable_grad():
            adj_para = torch.nn.Parameter(adj)
            F, _ = self.regressor(x, adj_para, flags, t).sum()
            # average gradients if model is an ensemble
            if isinstance(F, list):
                F = torch.stack(F, dim=0).mean(dim=0)
            F = F.sum()
            F.backward()
            score = adj_para.grad
            score = mask_adjs(score, flags)
        return score
