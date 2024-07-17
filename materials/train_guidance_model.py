import os
import json
import random
import shutil
import pickle
import itertools
import pandas as pd
import numpy as np
from datetime import datetime
from time import time, sleep
import warnings
from tqdm import tqdm

import torch
from torch.nn.functional import l1_loss, mse_loss, softplus
from torch import optim, Tensor, linspace

from edm.egnn_predictor.models import EGNN_predictor
from edm.equivariant_diffusion.utils import (
    remove_mean_with_mask,
    assert_correctly_masked,
    assert_mean_zero_with_mask,
)

from data.aromatic_dataloader import create_data_loaders, AromaticDataset
from models_edm import get_model, MyDataParallel
from cond_prediction.prediction_args import PredictionArgs
from utils.args_edm import Args_EDM

warnings.simplefilter(action="ignore", category=FutureWarning)

eps = 1e-6


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**3

    print("model size: {:.3f} GB".format(size_all_mb))


def sample_noisy_data(xh, node_mask, edge_mask, edm_model, t):
    """
    Sample noisy data at noise scale t.
    """

    gamma_t = edm_model.inflate_batch_array(edm_model.gamma(t), xh).to(xh.device)
    alpha_t = edm_model.alpha(gamma_t, xh)
    sigma_t = edm_model.sigma(gamma_t, xh)
    eps = edm_model.sample_combined_position_feature_noise(
        n_samples=xh.size(0), n_nodes=xh.size(1), node_mask=node_mask
    )
    z_t = alpha_t * xh + sigma_t * eps

    # print("\n\nNoise function shapes", gamma_t.shape, alpha_t.shape, sigma_t.shape, eps.shape, z_t.shape)
    return z_t, node_mask, edge_mask


def compute_loss(
    loss_fn,
    pred,
    target,
) -> Tensor:
    # use Gaussian likelihood loss

    if pred.isnan().any() or pred.isinf().any():
        print("NLL loss is NaN or inf")
        loss = torch.tensor(float("nan"))

    else:
        pred_mean = pred[:, : pred_args.num_targets]
        pred_vars = pred[:, pred_args.num_targets :]
        pred_vars = softplus(pred_vars)

        loss = loss_fn(pred_mean, target, pred_vars)

    return loss, pred_mean, pred_vars, target


def get_context_guided_reg(
    preds_f, preds_f_prior_mean, feature_prior, pred_args, ts
):
    """
    Compute the context-guided diffusion regularization term.
    
    Args:
        preds_f: torch.Tensor, shape (n_samples, 2 * num_targets).
            Predictions of the guidance model for the context points.
        preds_f_prior_mean: torch.Tensor, shape (n_samples, 2 * num_targets).
            Logits of the context model for the context points.
        feature_prior: torch.Tensor, shape (n_context_points, nf).
            Embeddings of the context points.
        pred_args: argparse.Namespace.
            Arguments for the predictor model.
        ts: torch.Tensor, shape (n_samples, 1).
            Diffusion time steps.
    """

    # make regularization hyperparameters time-dependent, this is an approximation
    # of the noise schedule used for training the diffusion model, using the exact
    # one leads to essentially identical results
    diag_scale = (
        pred_args.prior_likelihood_cov_diag / (1 + torch.exp(-7 * (ts - 0.5)))
        + pred_args.prior_likelihood_cov_diag / 10
    )
    cov_scale = (
        pred_args.prior_likelihood_cov_scale / (1 + torch.exp(7 * (ts - 0.5)))
        + pred_args.prior_likelihood_cov_scale / 10
    )

    feature_prior *= torch.sqrt(cov_scale)

    # construct the prior covariance matrix
    preds_f_prior_cov = torch.matmul(feature_prior, feature_prior.T)
    preds_f_prior_cov += (
        torch.eye(preds_f_prior_cov.shape[0]).to(pred_args.device) * diag_scale
    )

    if preds_f.isnan().any() or preds_f.isinf().any():
        print("preds_f is NaN or inf")
        fs_reg = torch.tensor(float("nan"))
    elif preds_f_prior_mean.isnan().any() or preds_f_prior_mean.isinf().any():
        print("preds_f_prior_mean is NaN or inf")
        fs_reg = torch.tensor(float("nan"))
    elif preds_f_prior_cov.isnan().any() or preds_f_prior_cov.isinf().any():
        print("preds_f_prior_cov is NaN or inf")
        fs_reg = torch.tensor(float("nan"))
    else:
        # check if the prior covariance matrix is positive definite
        L, info = torch.linalg.cholesky_ex(preds_f_prior_cov, upper=False)
        if info.any():
            print("Prior covariance matrix is not positive definite")
            fs_reg = torch.tensor(float("nan"))

        else:
            
            preds_f_prior_mean_means = preds_f_prior_mean[:, : pred_args.num_targets]
            preds_f_prior_mean_vars = preds_f_prior_mean[:, pred_args.num_targets :]
            preds_f_prior_mean_vars = softplus(preds_f_prior_mean_vars)

            # since the logits of the randomly initialized context model are centered
            # around zero, we investigated whether using them instead of a zero mean
            # vector leads to better results. However, this was not the case.
            # uncomment the following two lines to revert to the exact experimental 
            # setup described in the paper.
            # preds_f_prior_mean_means = torch.zeros_like(preds_f_prior_mean_means)
            # preds_f_prior_mean_vars = torch.zeros_like(preds_f_prior_mean_vars)

            means_likelihood = torch.distributions.MultivariateNormal(
                preds_f_prior_mean_means.T, preds_f_prior_cov
            )
            covs_likelihood = torch.distributions.MultivariateNormal(
                preds_f_prior_mean_vars.T + pred_args.log_prior_likelihood_var,
                preds_f_prior_cov,
            )

            preds_f_means = preds_f[:, : pred_args.num_targets]
            preds_f_vars = preds_f[:, pred_args.num_targets :]
            preds_f_vars = softplus(preds_f_vars)

            mean_logp = means_likelihood.log_prob(preds_f_means.T)
            cov_logp = covs_likelihood.log_prob(preds_f_vars.T)

            logps = torch.cat([mean_logp, cov_logp], dim=0)
            fs_reg = -logps.sum()

    return fs_reg


def get_param_norm(model, args):
    """
    Compute the norm of the model parameters.

    Args:
        model: nn.Module.

    Returns:
        torch.Tensor: Norm of the model parameters.
    """

    param_norm = torch.cat([p.view(-1) for p in model.parameters()]).square().sum()

    return param_norm


def get_grad_norm(model):
    grad_norm = (
        torch.cat(
            [
                p.grad.detach().data.view(-1)
                for p in model.parameters()
                if p.grad is not None
            ]
        )
        .square()
        .sum()
    )

    grad_norm = torch.sqrt(grad_norm)

    return grad_norm


def train_epoch(
    epoch,
    cond_predictor,
    cond_predictor_init,
    edm_model,
    dataloader,
    context_loader,
    optimizer,
    loss_fn,
    args,
):
    cond_predictor.train()

    start_time = time()
    loss_list, nll_loss_list, fs_reg_list, ps_reg_list, grad_norms = [], [], [], [], []

    for i, (xh, node_mask, edge_mask, y) in enumerate(dataloader):
        # uniformly sample time scales for guidance function training

        T = edm_args.diffusion_steps

        if args.reg_type == "fseb":
            # sample context point data
            context_xh, context_node_mask, context_edge_mask, _ = next(
                iter(context_loader)
            )

            # concat training samples with context points to speed up forward pass
            xh = torch.cat([xh, context_xh], dim=0)
            node_mask = torch.cat([node_mask, context_node_mask], dim=0)
            edge_mask = torch.cat([edge_mask, context_edge_mask], dim=0)

            t = torch.randint(0, T + 1, size=(len(xh), 1), device=xh.device).float() / T

            # run forward noising process
            with torch.no_grad():
                noised_xh, noised_node_mask, noised_edge_mask = sample_noisy_data(
                    xh, node_mask, edge_mask, edm_model, t
                )

            # get predictions for noised samples
            noised_preds, _ = cond_predictor(
                noised_xh, noised_node_mask, noised_edge_mask
            )

            # split back into training samples and context points
            xh = noised_xh[: args.batch_size]
            node_mask = noised_node_mask[: args.batch_size]
            edge_mask = noised_edge_mask[: args.batch_size]
            preds = noised_preds[: args.batch_size]
            ts = t[: args.batch_size]

            context_xh = noised_xh[args.batch_size :]
            context_node_mask = noised_node_mask[args.batch_size :]
            context_edge_mask = noised_edge_mask[args.batch_size :]
            context_preds = noised_preds[args.batch_size :]
            context_ts = t[args.batch_size :]

            # also get predictions for context points with initial model
            with torch.no_grad():
                context_batch_logits, context_batch_features = cond_predictor_init(
                    context_xh, context_node_mask, context_edge_mask
                )

        else:
            # only apply forward noising process to training samples
            t = torch.randint(0, T + 1, size=(len(xh), 1), device=xh.device).float() / T
            with torch.no_grad():
                xh, node_mask, edge_mask = sample_noisy_data(
                    xh, node_mask, edge_mask, edm_model, t
                )
            preds, _ = cond_predictor(xh, node_mask, edge_mask)
            context_batch_logits, context_batch_features = None, None

        # compute training set loss

        nll_loss, _, _, _ = compute_loss(loss_fn, preds, y)

        if nll_loss.isnan().any():
            return 1

        nll_loss_list.append(nll_loss.item())

        # compute regularisation term

        reg_scale = args.batch_size * len(dataloader) * args.num_targets

        if args.reg_type == "fseb":
            # get function-space regularisation term
            fs_reg = get_context_guided_reg(
                preds_f=context_preds,
                preds_f_prior_mean=context_batch_logits,
                feature_prior=context_batch_features,
                pred_args=args,
                ts=context_ts,
            )
            fs_reg /= reg_scale

        else:
            fs_reg = torch.zeros_like(nll_loss)

        if args.prior_var:
            ps_reg = 1 / (2 * args.prior_var) * get_param_norm(cond_predictor, args)
            ps_reg /= reg_scale

        else:
            ps_reg = torch.zeros_like(nll_loss)

        # save regularisation terms

        if fs_reg.isnan().any() or ps_reg.isnan().any():
            return 1

        fs_reg_list.append(fs_reg.item())
        ps_reg_list.append(ps_reg.item())

        # rescale regularisation term by number of training batches
        # and add to loss

        loss = nll_loss + fs_reg + ps_reg
        loss_list.append(loss.item())

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log grad norm
        grad_norms.append(get_grad_norm(cond_predictor).item())
        torch.nn.utils.clip_grad_norm_(cond_predictor.parameters(), 50)

    print(
        f"[{epoch}|train] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
        f"NLL loss: {np.mean(nll_loss_list):.4f}+-{np.std(nll_loss_list):.4f}, "
        f"FS reg loss: {np.mean(fs_reg_list):.4f}+-{np.std(fs_reg_list):.4f}, "
        f"PS reg loss: {np.mean(ps_reg_list):.4f}+-{np.std(ps_reg_list):.4f}, "
        f"mean(max) grad norm: {np.mean(grad_norms):.4f}+-{np.std(grad_norms):.4f} ({np.max(grad_norms):.4f}), "
        # f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
        f" in {int(time()-start_time)} secs"
    )
    sleep(0.01)

    return 0


def val_epoch(
    tag,
    epoch,
    cond_predictor,
    edm_model,
    dataloader,
    loss_fn,
    args,
    return_preds=False,
):
    T = edm_args.diffusion_steps
    eval_times = [t / T for t in range(0, T + 1, int(T / 20))]

    cond_predictor.eval()
    with torch.no_grad():

        start_time = time()
        loss_list, pred_mean_dict, pred_logvar_dict, pred_target_dict = (
            [],
            dict(),
            dict(),
            dict(),
        )
        xh, node_mask, edge_mask, y = dataloader.dataset.tensors

        for t in eval_times:

            # noise all datapoints to current time scale
            xh_noised, node_mask_noised, edge_mask_noised = sample_noisy_data(
                xh,
                node_mask,
                edge_mask,
                edm_model,
                torch.full((len(xh), 1), fill_value=t, device=xh.device),
            )

            preds, _ = cond_predictor(xh_noised, node_mask_noised, edge_mask_noised)
            loss, pred_mean, pred_logvar, pred_target = compute_loss(
                loss_fn,
                preds,
                y,
            )

            loss_list.append(loss.item())

            if return_preds:
                pred_mean_dict[t] = pred_mean.cpu().numpy()
                pred_logvar_dict[t] = pred_logvar.cpu().numpy()
                pred_target_dict[t] = pred_target.cpu().numpy()

        print(
            f"[{epoch}|{tag}] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
            # f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
            f" in {int(time() - start_time)} secs"
        )
        sleep(0.01)

    return np.mean(loss_list), pred_mean_dict, pred_logvar_dict, pred_target_dict


def get_cond_predictor_model(args, dataset: AromaticDataset, num_node_features):
    cond_predictor = EGNN_predictor(
        in_nf=num_node_features,
        device=args.device,
        hidden_nf=args.nf,
        out_nf=args.num_targets * 2,  # mean and logvar
        act_fn=torch.nn.SiLU(),
        n_layers=args.n_layers,
        recurrent=True,
        tanh=args.tanh,
        attention=args.attention,
        condition_time=True,
        coords_range=args.coords_range,
    )

    if args.dp:  # and torch.cuda.device_count() > 1:
        cond_predictor = MyDataParallel(cond_predictor)
    if args.restore is not None:
        model_state_dict = torch.load(
            args.exp_dir + "/model.pt", map_location=args.device
        )
        cond_predictor.load_state_dict(model_state_dict)

        print("Loaded model from", args.exp_dir + "/model.pt")
    return cond_predictor


def main(pred_args, edm_args, return_preds=False):
    # Prepare data
    (
        train_loader,
        val_loader,
        test_loader,
        context_loader,
        num_node_features,
    ) = create_data_loaders(pred_args)

    print("\n\nTrain size:", len(train_loader.dataset))
    print("Val size:", len(val_loader.dataset))
    print("Test size:", len(test_loader.dataset))
    print("Context size:", len(context_loader.dataset))

    # create diffusion model

    edm_model, nodes_dist, prop_dist = get_model(
        edm_args, train_loader, in_node_nf=num_node_features
    )

    if pred_args.reg_type == "fseb":
        cond_predictor_init = get_cond_predictor_model(
            pred_args, train_loader.dataset, num_node_features
        )
        cond_predictor_init.eval()

    else:
        cond_predictor_init = None

    cond_predictor = get_cond_predictor_model(
        pred_args, train_loader.dataset, num_node_features
    )

    if "pretrain" in pred_args.reg_type:
        print("Loading pretrained model")
        cond_predictor.load_state_dict(torch.load("pretrained_model.pth"))

    print("\n\nCond predictor model size:")
    get_model_size(cond_predictor)
    if pred_args.reg_type == "fseb":
        print("Cond predictor init model size:")
        get_model_size(cond_predictor_init)
    print("EDM model size:")
    get_model_size(edm_model)

    optimizer = optim.Adam(
        cond_predictor.parameters(),
        lr=pred_args.lr,
        amsgrad=True,  # weight_decay=pred_args.weight_decay
    )

    loss_fn = torch.nn.GaussianNLLLoss(full=True, reduction="mean")

    # Save path

    # Run training with early stopping
    print("Begin training")
    best_val_loss = 1e9
    best_epoch = 0
    early_stopping_counter = 0

    for epoch in range(pred_args.num_epochs):
        return_code = train_epoch(
            epoch,
            cond_predictor,
            cond_predictor_init,
            edm_model,
            train_loader,
            context_loader,
            optimizer,
            loss_fn,
            pred_args,
        )

        if return_code == 1:
            print(
                "Exiting training due to NaN loss or non-positive definite prior covariance matrix"
            )
            break

        val_loss, _, _, _ = val_epoch(
            "val",
            epoch,
            cond_predictor,
            edm_model,
            val_loader,
            loss_fn,
            pred_args,
            return_preds=False,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stopping_counter = 0
            torch.save(cond_predictor.state_dict(), pred_args.exp_dir + "/model.pt")
        else:
            early_stopping_counter += 1
            if early_stopping_counter > pred_args.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch} with val loss {val_loss:.4f} and best val loss {best_val_loss:.4f}"
                )
                break

    # re-load best model, if possible

    if os.path.isfile(pred_args.exp_dir + "/model.pt"):
        print(f"{best_epoch=}, {best_val_loss=:.4f}")
        cond_predictor.load_state_dict(torch.load(pred_args.exp_dir + "/model.pt"))

        # get val and test loss with best model

        val_loss, val_means, val_logvars, val_targets = val_epoch(
            "val",
            epoch,
            cond_predictor,
            edm_model,
            val_loader,
            loss_fn,
            pred_args,
            return_preds=return_preds,
        )

        if return_preds:

            train_loss, train_means, train_logvars, train_targets = val_epoch(
                "train",
                epoch,
                cond_predictor,
                edm_model,
                train_loader,
                loss_fn,
                pred_args,
                return_preds=return_preds,
            )

            test_loss, test_means, test_logvars, test_targets = val_epoch(
                "test",
                epoch,
                cond_predictor,
                edm_model,
                test_loader,
                loss_fn,
                pred_args,
                return_preds=return_preds,
            )

            context_loss, context_means, context_logvars, context_targets = val_epoch(
                "context",
                epoch,
                cond_predictor,
                edm_model,
                context_loader,
                loss_fn,
                pred_args,
                return_preds=return_preds,
            )

        else:
            train_loss, train_means, train_logvars, train_targets = (
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            )
            test_loss, test_means, test_logvars, test_targets = (
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            )
            context_loss, context_means, context_logvars, context_targets = (
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            )

        results = {
            "return_code": return_code,
            "best_loss": best_val_loss,
            "best_epoch": best_epoch,
            "val_loss": val_loss,
            "val_means": val_means,
            "val_logvars": val_logvars,
            "val_targets": val_targets,
            "test_loss": test_loss,
            "test_means": test_means,
            "test_logvars": test_logvars,
            "test_targets": test_targets,
            "train_loss": train_loss,
            "train_means": train_means,
            "train_logvars": train_logvars,
            "train_targets": train_targets,
            "context_loss": context_loss,
            "context_means": context_means,
            "context_logvars": context_logvars,
            "context_targets": context_targets,
        }

    else:
        print(f"No model file {pred_args.exp_dir + '/model.pt'} exists. Skipping.")
        results = {
            "return_code": return_code,
            "best_loss": float("nan"),
            "best_epoch": float("nan"),
            "val_loss": float("nan"),
            "val_means": float("nan"),
            "val_logvars": float("nan"),
            "val_targets": float("nan"),
            "test_loss": float("nan"),
            "test_means": float("nan"),
            "test_logvars": float("nan"),
            "test_targets": float("nan"),
            "train_loss": float("nan"),
            "train_means": float("nan"),
            "train_logvars": float("nan"),
            "train_targets": float("nan"),
            "context_loss": float("nan"),
            "context_means": float("nan"),
            "context_logvars": float("nan"),
            "context_targets": float("nan"),
        }

    return results


if __name__ == "__main__":
    # scatter functions used in EGNN code cannot be made deterministic :(
    # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
    # torch.use_deterministic_algorithms(True)

    pred_args = PredictionArgs().parse_args()
    edm_args = Args_EDM().parse_args([])
    edm_args.dp = False

    # map integer batch job SLURM ID to arguments

    split_types = ["random_split", "cluster_split"]
    context_set_types = ["rings_10", "all"]
    reg_type = ["fseb", "ps", "none"]
    arg_id_list = [
        (s, c, r)
        for s, c, r in itertools.product(split_types, context_set_types, reg_type)
    ]

    # context set does not matter for parameter space regularization
    arg_id_list.remove(("cluster_split", "rings_10", "ps"))
    arg_id_list.remove(("random_split", "rings_10", "ps"))
    arg_id_list.remove(("cluster_split", "rings_10", "none"))
    arg_id_list.remove(("random_split", "rings_10", "none"))

    arg_id_list.append(("cluster_split", "all", "ps_pretrain"))

    pred_args.split, pred_args.context_set, pred_args.reg_type = arg_id_list[
        pred_args.arg_id
    ]

    print(f"\n\nSplit: {pred_args.split}")
    print(f"Context set: {pred_args.context_set}")
    print(f"Reg type: {pred_args.reg_type}")

    # map hyperparameter ID to hyperparameter values

    if pred_args.reg_type == "fseb":
        hyp_prior_likelihood_cov_scale = [1e-2, 1e-1, 1e0, 1e1, 1e-2]
        hyp_prior_likelihood_cov_diag = [1e-2, 1e-1, 1e0, 1e1, 1e-2]
        hyp_log_prior_likelihood_var = [1]
        hyp_prior_var = [0]
        n_context_points = [16, 64, 256]

    elif pred_args.reg_type == "ps" or pred_args.reg_type == "ps_pretrain":
        hyp_prior_likelihood_cov_scale = [0]
        hyp_prior_likelihood_cov_diag = [0]
        hyp_log_prior_likelihood_var = [0]
        hyp_prior_var = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e-3]
        n_context_points = [pred_args.batch_size]

    elif pred_args.reg_type == "none":
        hyp_prior_likelihood_cov_scale = [0]
        hyp_prior_likelihood_cov_diag = [0]
        hyp_log_prior_likelihood_var = [0]
        hyp_prior_var = [0]
        n_context_points = [pred_args.batch_size]

    else:
        raise ValueError(f"Unknown reg_type: {pred_args.reg_type}")

    hyper_id_list = [
        {
            "prior_var": pv,
            "prior_likelihood_cov_scale": plcs,
            "prior_likelihood_cov_diag": plcd,
            "log_prior_likelihood_var": lplv,
            "n_context_points": ncp,
        }
        for pv, plcs, plcd, lplv, ncp in itertools.product(
            hyp_prior_var,
            hyp_prior_likelihood_cov_scale,
            hyp_prior_likelihood_cov_diag,
            hyp_log_prior_likelihood_var,
            n_context_points,
        )
        if not (
            plcs != 0 and plcd / plcs < 1e-3
        )  # runs with low diag to scale ration always fail
    ]

    print("Length of hyperparameter grid:", len(hyper_id_list))

    if pred_args.hyper_id < len(hyper_id_list):
        # if hyper_id is valid, use it

        hypers = hyper_id_list[pred_args.hyper_id]
        pred_args.prior_var = hypers["prior_var"]
        pred_args.prior_likelihood_cov_scale = hypers["prior_likelihood_cov_scale"]
        pred_args.prior_likelihood_cov_diag = hypers["prior_likelihood_cov_diag"]
        pred_args.log_prior_likelihood_var = hypers["log_prior_likelihood_var"]
        pred_args.n_context_points = hypers["n_context_points"]

        print(f"prior_var: {pred_args.prior_var}")
        print(f"prior_likelihood_cov_scale: {pred_args.prior_likelihood_cov_scale}")
        print(f"prior_likelihood_cov_diag: {pred_args.prior_likelihood_cov_diag}")
        print(f"log_prior_likelihood_var: {pred_args.log_prior_likelihood_var}")
        print(f"n_context_points: {pred_args.n_context_points}\n\n")

        pred_args.run_name = f"{pred_args.arg_id}_{pred_args.hyper_id}"
        pred_args.exp_dir = (
            f"{pred_args.save_dir}/{pred_args.name}/run_logs/{pred_args.run_name}"
        )

        results_path = f"{pred_args.save_dir}/{pred_args.name}/{pred_args.run_name}.pkl"

        if os.path.isfile(results_path):
            print(f"Results file {results_path} already exists. Skipping.")
            exit()

        # Create model directory
        if not os.path.isdir(pred_args.exp_dir):
            os.makedirs(pred_args.exp_dir)

        with open(pred_args.exp_dir + "/args.txt", "w") as f:
            json.dump(pred_args.__dict__, f, indent=2)

        # Automatically choose GPU if available
        pred_args.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        edm_args.device = pred_args.device
        print("CUDA:", torch.cuda.is_available())
        print("\n\nPred args:", pred_args)
        print("\n\nEDM args:", edm_args)

        # set random seed

        torch.manual_seed(pred_args.seed)
        np.random.seed(pred_args.seed)
        random.seed(pred_args.seed)

        results = main(pred_args, edm_args, return_preds=False)

        # Save log dict as pickle file

        log_dict = {
            "args": pred_args.__dict__,
            "split": pred_args.split,
            "context_set": pred_args.context_set,
            "reg_type": pred_args.reg_type,
            **hypers,
            **results,
        }

        with open(results_path, "wb") as f:
            pickle.dump(log_dict, f)

    else:
        # if hyper_id exceeds length of hyperparameter grid,
        # but final rerun result does not exist, then rerun

        for i in range(pred_args.seed + 1, pred_args.seed + 11):

            rerun_path = f"{pred_args.save_dir}/{pred_args.name}/{pred_args.arg_id}_final_{i}.pkl"

            if not os.path.isfile(rerun_path):
                # read in best hyperparameters

                print(f"\n\n\nRerun {i} of {pred_args.seed+10}")
                print("\n\n\nSearching the following files for best hyperparameters:")

                result_files = [
                    f
                    for f in os.listdir(f"{pred_args.save_dir}/{pred_args.name}")
                    if f.startswith(str(pred_args.arg_id))
                    and f.endswith(".pkl")
                    and "final" not in f
                ]
                results = []

                for f in result_files:
                    print(f)
                    with open(
                        f"{pred_args.save_dir}/{pred_args.name}/{f}", "rb"
                    ) as file:
                        results.append(pickle.load(file))

                results = pd.DataFrame(results).drop(columns=["args"])
                best_result = results.loc[results["val_loss"].idxmin()]

                print("\n\nSelected best hyperparameters:")
                print(best_result)

                # set hyperparameters

                pred_args.prior_var = best_result["prior_var"]
                pred_args.prior_likelihood_cov_scale = float(
                    best_result["prior_likelihood_cov_scale"]
                )
                pred_args.prior_likelihood_cov_diag = float(
                    best_result["prior_likelihood_cov_diag"]
                )
                pred_args.log_prior_likelihood_var = float(
                    best_result["log_prior_likelihood_var"]
                )
                pred_args.n_context_points = int(best_result["n_context_points"])

                # Automatically choose GPU if available
                pred_args.device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                edm_args.device = pred_args.device
                print("CUDA:", torch.cuda.is_available())
                print("\n\nPred args:", pred_args)
                print("\n\nEDM args:", edm_args)

                pred_args.run_name = (
                    f"{pred_args.arg_id}_{pred_args.hyper_id}_rerun_{i}"
                )
                pred_args.exp_dir = f"{pred_args.save_dir}/{pred_args.name}/run_logs/{pred_args.run_name}"

                # Create model directory
                if not os.path.isdir(pred_args.exp_dir):
                    os.makedirs(pred_args.exp_dir)

                # reset random seed

                torch.manual_seed(i)
                np.random.seed(i)
                random.seed(i)

                print("CUDA:", torch.cuda.is_available())
                # print("Pred args:", pred_args)

                results = main(pred_args, edm_args, return_preds=True)

                log_dict = {
                    "args": pred_args.__dict__,
                    "rerun_iteraton": i,
                    "split": pred_args.split,
                    "context_set": pred_args.context_set,
                    "reg_type": pred_args.reg_type,
                    "prior_var": pred_args.prior_var,
                    "prior_likelihood_cov_scale": pred_args.prior_likelihood_cov_scale,
                    "prior_likelihood_cov_diag": pred_args.prior_likelihood_cov_diag,
                    "log_prior_likelihood_var": pred_args.log_prior_likelihood_var,
                    "n_context_points": pred_args.n_context_points,
                    **results,
                }

                with open(rerun_path, "wb") as f:
                    pickle.dump(log_dict, f)

            else:
                print(
                    f"Hyper ID {pred_args.hyper_id} exceeded legnth of specified hyperparameter grid ({len(hyper_id_list)}) and results file {rerun_path} already exists. Skipping."
                )
