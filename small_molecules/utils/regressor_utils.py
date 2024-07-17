import torch
import random
import pandas as pd
from torch import autograd
from torch.autograd import Function
from torch.nn.functional import gaussian_nll_loss, softplus

from utils.graph_utils import node_flags, mask_x, mask_adjs, gen_noise
from models.regressor import Regressor, RegressorEnsemble, get_regressor_fn
from utils.loader import load_sde

# --------------------------- domain adaptation --------------------------- #

def split_into_groups(g):
    """
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - groups (Tensor): Unique groups present in g
        - group_indices (list): List of Tensors, where the i-th tensor is the indices of the
                                elements of g that equal groups[i].
                                Has the same length as len(groups).
        - unique_counts (Tensor): Counts of each element in groups.
                                 Has the same length as len(groups).
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(
            torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts


class GradientReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def dann_regularizer(groups, features, dann_model, dann_alpha=1):
    """
    Adapted from https://github.com/tencent-ailab/DrugOOD/drugood/models/algorithms/dann.py
    """

    _features = GradientReverseLayerF.apply(features, dann_alpha)
    dann_loss = dann_model.loss(_features, groups)
    return dann_loss

def coral_regularizer(groups, features, coral_ratio=1):

    """
    Adapted from https://github.com/tencent-ailab/DrugOOD/drugood/models/algorithms/coral.py
    """

    unique_groups, group_indices, _ = split_into_groups(groups)
    coral_penalty = []
    n_groups_per_batch = unique_groups.numel()
    for i_group in range(n_groups_per_batch):
        for j_group in range(i_group + 1, n_groups_per_batch):

            x = features[group_indices[i_group]]
            y = features[group_indices[j_group]]

            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            coral_penalty.append(mean_diff + cova_diff)

    if len(coral_penalty) > 1:
        coral_penalty = torch.vstack(coral_penalty).mean()
    elif len(coral_penalty) == 1:
        coral_penalty = coral_penalty[0]

    return coral_penalty * coral_ratio


def domain_confusion(class_or_conf, groups, features, dann_model, domain_confusion_hyper=1):
    """
    Adapted from https://arxiv.org/pdf/1510.02192.pdf

    Args:
        - class_or_conf (str): whether to optimize for 
        - groups (Tensor): Vector of groups
        - features (Tensor): Features to be domain-confused
        - dann_model (nn.Module): Domain adaptation model
        - domain_confusion_hyper (float): Hyperparameter for domain adaptation
    """

    unique_groups, group_indices, _ = split_into_groups(groups)

    if class_or_conf:
        # train domain classification model
        dann_loss = dann_model.loss(features, groups)
    else:
        # maximize cross-entropy of predictions and uniform distribution
        dann_preds = torch.nn.functional.softmax(dann_model(features), dim=-1)
        dann_loss = -torch.log(dann_preds).mean()

    return dann_loss * domain_confusion_hyper


# --------------------------- our regulariser --------------------------- #


def get_context_logit_mean(
    config,
    context_model,
    context_loader,
):

    logits = []

    context_model.eval()

    with torch.no_grad():

        sde_x = load_sde(config.sde.x)
        sde_adj = load_sde(config.sde.adj)
        eps = config.train.eps
        regressor_fn = get_regressor_fn(sde_adj, context_model)

        for batch in context_loader:

            x, adj, _ = load_regressor_batch(batch)
            flags = node_flags(adj)
            t = torch.rand(adj.shape[0], device=adj.device) * (sde_adj.T - eps) + eps

            z_x = gen_noise(x, flags, sym=False)
            mean_x, std_x = sde_x.marginal_prob(x, t)
            perturbed_x = mean_x + std_x[:, None, None] * z_x
            perturbed_x = mask_x(perturbed_x, flags)

            z_adj = gen_noise(adj, flags, sym=True)
            mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
            perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
            perturbed_adj = mask_adjs(perturbed_adj, flags)

            pred, _ = regressor_fn(perturbed_x, perturbed_adj, flags, t)

            logits.append(pred[:, 0].squeeze())

    logits = torch.cat(logits)

    return logits.mean()


def get_context_guided_reg_nomve(
    preds_f,
    preds_f_prior_mean,
    feature_prior,
    ts,
    prior_likelihood_cov_diag,
    prior_likelihood_cov_scale,
    train_y_mean,
    context_logits_mean,
    device,
):
    """
    Compute the empirical prior density of the model parameters.

    Args:
        trained_model: nn.Module.
        init_model: nn.Module.
        x: torch.Tensor.
        pred_args: PredictionArgs.

    Returns:
        torch.Tensor: Empirical prior density of the model parameters.
    """

    # scale prediction regularisation to make it weaker at larger noise scales
    diag_scale = (
        prior_likelihood_cov_diag / (1 + torch.exp(-7 * (ts - 0.5))) + prior_likelihood_cov_diag / 10
        # torch.ones_like(ts) * prior_likelihood_cov_diag
        # prior_likelihood_cov_diag / 10 + ts * (prior_likelihood_cov_diag - prior_likelihood_cov_diag / 10)
    ).unsqueeze(-1)

    cov_scale = (
        prior_likelihood_cov_scale / (1 + torch.exp(7 * (ts - 0.5))) + prior_likelihood_cov_scale / 10
        # exit
        # prior_likelihood_cov_scale + ts * (prior_likelihood_cov_scale / 10 - prior_likelihood_cov_scale)
    ).unsqueeze(-1)

    feature_prior = feature_prior * torch.sqrt(cov_scale)

    # construct the prior covariance matrix
    preds_f_prior_cov = torch.matmul(feature_prior, feature_prior.T)

    # preds_f_prior_cov += (
    #    torch.ones_like(preds_f_prior_cov) * cov_scale
    # )

    preds_f_prior_cov = preds_f_prior_cov + (
        torch.eye(preds_f_prior_cov.shape[0]).to(f"cuda:{device[0]}") * diag_scale
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
        # print("FS-EB shapes", preds_f_prior_mean.shape, preds_f_prior_cov.shape, preds_f.shape)

        # check if the prior covariance matrix is positive definite
        with torch.no_grad():
            L, info = torch.linalg.cholesky_ex(preds_f_prior_cov, upper=False)
        if info.any():
            print("Prior covariance matrix is not positive definite")
            fs_reg = torch.tensor(float("nan"))

        else:

            means_likelihood = torch.distributions.MultivariateNormal(
                preds_f_prior_mean.T - (context_logits_mean - train_y_mean), preds_f_prior_cov
            )
            mean_logp = means_likelihood.log_prob(preds_f.T)

            fs_reg = -mean_logp.sum()

    return fs_reg


def get_context_guided_reg(
    preds_f,
    preds_f_prior_mean,
    feature_prior,
    ts,
    prior_likelihood_cov_diag,
    prior_likelihood_cov_scale,
    train_y_mean,
    context_logits_mean,
    device,
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
        ts: torch.Tensor, shape (n_samples, 1).
            Diffusion time steps.
        prior_likelihood_cov_diag: float.
            Diagonal offset hyperparameter.
        prior_likelihood_cov_scale: float.
            Covariance scale hyperparameter.
        train_y_mean: float.
            Mean of the training labels.
        context_logits_mean: float.
            Mean of the logits of the randomly initialized model.
        device: str.
            Device to use.
    """


    # make regularization hyperparameters time-dependent, this is an approximation
    # of the noise schedule used for training the diffusion model, using the exact
    # one leads to essentially identical results
    diag_scale = (
        prior_likelihood_cov_diag / (1 + torch.exp(-7 * (ts - 0.5))) + prior_likelihood_cov_diag / 10
    ).unsqueeze(-1)

    cov_scale = (
        prior_likelihood_cov_scale / (1 + torch.exp(7 * (ts - 0.5))) + prior_likelihood_cov_scale / 10
    ).unsqueeze(-1)

    feature_prior = feature_prior * torch.sqrt(cov_scale)

    # construct the prior covariance matrix
    preds_f_prior_cov = torch.matmul(feature_prior, feature_prior.T)

    preds_f_prior_cov = preds_f_prior_cov + (
        torch.eye(preds_f_prior_cov.shape[0]).to(f"cuda:{device[0]}") * diag_scale
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
        # print("FS-EB shapes", preds_f_prior_mean.shape, preds_f_prior_cov.shape, preds_f.shape)

        # check if the prior covariance matrix is positive definite
        with torch.no_grad():
            L, info = torch.linalg.cholesky_ex(preds_f_prior_cov, upper=False)
        if info.any():
            print("Prior covariance matrix is not positive definite")
            fs_reg = torch.tensor(float("nan"))

        else:

            # since the logits of the randomly initialized context model are centered
            # around zero, we investigated whether using them instead of a zero mean
            # vector leads to better results. However, this was not the case.
            # uncomment the following two lines to revert to the exact experimental 
            # setup described in the paper.
            # preds_f_prior_mean_means = torch.zeros_like(preds_f_prior_mean_means)
            # preds_f_prior_mean_vars = torch.zeros_like(preds_f_prior_mean_vars)

            preds_f_prior_mean_means = preds_f_prior_mean[:, 0].unsqueeze(-1)
            preds_f_prior_mean_vars = softplus(preds_f_prior_mean[:, 1])#.unsqueeze(-1)

            means_likelihood = torch.distributions.MultivariateNormal(
                preds_f_prior_mean_means.T - (context_logits_mean - train_y_mean), preds_f_prior_cov
            )
            covs_likelihood = torch.distributions.MultivariateNormal(
                torch.ones_like(preds_f_prior_mean_vars), preds_f_prior_cov,
            )

            preds_f_means = preds_f[:, 0].unsqueeze(-1)
            preds_f_vars = softplus(preds_f[:, 1]).unsqueeze(-1)

            mean_logp = means_likelihood.log_prob(preds_f_means.T)
            cov_logp = covs_likelihood.log_prob(preds_f_vars.T)

            logps = torch.cat([mean_logp, cov_logp], dim=0)
            # this is a sum over n_context_points and n_targets
            fs_reg = -logps.sum()


    return fs_reg

# --------------------------- PS regulariser --------------------------- #


def get_param_norm(model):

    param_norm = torch.cat([p.view(-1) for p in model.parameters()]).square().sum()

    return param_norm


def load_regressor_params(config):
    config_m = config.model
    params = {
        "max_node_num": config.data.max_node_num,
        "max_feat_num": config.data.max_feat_num,
        "depth": config_m.depth,
        "nhid": config_m.nhid,
        "dropout": config_m.dropout,
    }

    return params


def load_regressor(params):
    params_ = params.copy()

    if "ensemble_size" in params_.keys() and params_["ensemble_size"] > 1:
        model = RegressorEnsemble(**params_)
        print("Using deep ensemble model.")
    else:
        model = Regressor(**params_)
        print("Using single model.")

    return model


def load_regressor_optimizer(params, config_train, device):

    model = load_regressor(params).to(f"cuda:{device[0]}")

    if "pretrained" in config_train.reg_type:
        print("Loading pretrained model")
        pretrained_model_state = torch.load("checkpoints/ZINC250k/pretrained_model_reinit.pth")
        model.load_state_dict(pretrained_model_state)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config_train.lr, weight_decay=config_train.weight_decay
    )
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config_train.lr_decay
        )

    return model, optimizer, scheduler


def load_regressor_batch(batch):
    x_b = batch[0]  # .to(f"cuda:{device[0]}")
    adj_b = batch[1]  # .to(f"cuda:{device[0]}")

    if len(batch) == 3:
        label_b = batch[2].unsqueeze(-1)  # .to(f"cuda:{device[0]}")
    else:
        label_b = None

    return x_b, adj_b, label_b


def load_regressor_loss_fn(config, device):

    sde_x = load_sde(config.sde.x)
    sde_adj = load_sde(config.sde.adj)
    eps = config.train.eps

    def loss_fn(
        model,
        x,
        adj,
        labels,
        context_model,
        x_context,
        adj_context,
        reg_type,
        train_y_mean,
        context_logits_mean,
    ):

        # concatenate data and context points to pass them throught the
        # nosing process and the regressor function together

        if context_model is not None and reg_type in ["fseb", "coral", "dann", "domain_confusion"]:

            n_data, n_context = x.shape[0], x_context.shape[0]
            x = torch.cat([x, x_context], dim=0)
            adj = torch.cat([adj, adj_context], dim=0)
            groups = torch.cat([torch.zeros(n_data, dtype=torch.long), torch.ones(n_context, dtype=torch.long)]).to(x.device)

            if reg_type == "fseb":
                context_fn = get_regressor_fn(sde_adj, context_model)

        regressor_fn = get_regressor_fn(sde_adj, model)
        flags = node_flags(adj)
        t = torch.rand(adj.shape[0], device=adj.device) * (sde_adj.T - eps) + eps

        z_x = gen_noise(x, flags, sym=False)
        mean_x, std_x = sde_x.marginal_prob(x, t)
        perturbed_x = mean_x + std_x[:, None, None] * z_x
        perturbed_x = mask_x(perturbed_x, flags)

        z_adj = gen_noise(adj, flags, sym=True)
        mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
        perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
        perturbed_adj = mask_adjs(perturbed_adj, flags)

        if context_model is not None and reg_type in ["fseb", "coral", "dann", "domain_confusion"]:

            preds, embeds = regressor_fn(perturbed_x, perturbed_adj, flags, t)

            # split the data and context points

            perturbed_x_context = perturbed_x[n_data:]
            perturbed_adj_context = perturbed_adj[n_data:]

            flags_data, flags_context = flags[:n_data], flags[n_data:]
            t_data, t_context = t[:n_data], t[n_data:]
            pred, pred_context = preds[:n_data], preds[n_data:]

            if reg_type == "fseb":

                # pass context points through the context model

                with torch.no_grad():
                    context_logits, context_features = context_fn(
                        perturbed_x_context,
                        perturbed_adj_context,
                        flags_context,
                        t_context,
                    )

                embeds_pred, embeds_context = embeds[:n_data], embeds[n_data:]

                # compute the FSEB regulariser
                fseb_reg = get_context_guided_reg(
                    preds_f=pred_context,
                    preds_f_prior_mean=context_logits,
                    feature_prior=context_features, #TODO: check out embeds_context
                    ts=t_context,
                    prior_likelihood_cov_diag=config.train.prior_likelihood_cov_diag,
                    prior_likelihood_cov_scale=config.train.prior_likelihood_cov_scale,
                    train_y_mean=train_y_mean,
                    context_logits_mean=context_logits_mean,
                    device=device,
                )

            elif reg_type == "coral":
                print("Using coral")

                fseb_reg = coral_regularizer(
                    groups=groups,
                    features=embeds,
                    coral_ratio=config.train.coral_ratio,
                )

            elif reg_type == "dann":
                fseb_reg = dann_regularizer(
                    groups=groups,
                    features=embeds,
                    dann_model=context_model,
                    dann_alpha=config.train.dann_alpha,
                )

            elif reg_type == "domain_confusion":
                domain_confusion_switch = bool(random.getrandbits(1))
                fseb_reg = domain_confusion(
                    class_or_conf=domain_confusion_switch,
                    groups=groups,
                    features=embeds,
                    dann_model=context_model,
                    domain_confusion_hyper=config.train.domain_confusion_hyper,
                )
        else:

            pred, _ = regressor_fn(perturbed_x, perturbed_adj, flags, t)
            fseb_reg = torch.zeros(1, device=f"cuda:{device[0]}")

        if reg_type == "ps":
            ps_reg = 1 / (2 * config.train.ps_var) * get_param_norm(model)

        else:
            ps_reg = torch.zeros_like(fseb_reg)

        if "ensemble" in reg_type:
            
            #loss = [(pr - labels).pow(2).mean() for pr in pred]

            loss = [
                gaussian_nll_loss(labels, pr[:, 0], softplus(pr[:, 1]), full=True, reduction="mean")
                for pr in pred
            ]

        else:
            # compute the loss
            #loss = (pred - labels).pow(2).mean()

            loss = gaussian_nll_loss(labels, pred[:, 0], softplus(pred[:, 1]), full=True, reduction="mean")

        with torch.no_grad():
            
            if isinstance(pred, list):
                pred = pred[0]

            df = pd.DataFrame()
            df["pred"] = pred[:, 0].cpu().detach().numpy().squeeze()
            df["labels"] = labels.cpu().detach().numpy().squeeze()
            corr = df.corr()["pred"]["labels"]

        return loss, fseb_reg, ps_reg, corr

    return loss_fn


def old_load_regressor_loss_fn(config):
    sde_x = load_sde(config.sde.x)
    sde_adj = load_sde(config.sde.adj)
    eps = config.train.eps

    def loss_fn(model, x, adj, labels):
        regressor_fn = get_regressor_fn(sde_adj, model)
        flags = node_flags(adj)
        t = torch.rand(adj.shape[0], device=adj.device) * (sde_adj.T - eps) + eps

        z_x = gen_noise(x, flags, sym=False)
        mean_x, std_x = sde_x.marginal_prob(x, t)
        perturbed_x = mean_x + std_x[:, None, None] * z_x
        perturbed_x = mask_x(perturbed_x, flags)

        z_adj = gen_noise(adj, flags, sym=True)
        mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
        perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
        perturbed_adj = mask_adjs(perturbed_adj, flags)

        pred = regressor_fn(perturbed_x, perturbed_adj, flags, t)
        loss = (pred - labels).pow(2).mean()

        with torch.no_grad():
            df = pd.DataFrame()
            df["pred"] = pred.cpu().detach().numpy().squeeze()
            df["labels"] = labels.cpu().detach().numpy().squeeze()
            corr = df.corr()["pred"]["labels"]

        return loss, corr

    return loss_fn


def load_regressor_from_ckpt(params, state_dict, device):
    model = load_regressor(params)
    model.load_state_dict(state_dict)
    model = model.to(f"cuda:{device[0]}")

    return model


def load_regressor_ckpt(config, device):
    ckpt_dict = {}
    path = f"./checkpoints/{config.data.data}/{config.model.prop.ckpt}.pth"
    ckpt = torch.load(path, map_location=f"cuda:{device[0]}")
    print(f"{path} loaded")
    ckpt_dict["prop"] = {
        "config": ckpt["model_config"],
        "params": ckpt["params"],
        "state_dict": ckpt["state_dict"],
    }
    ckpt_dict["prop"]["config"]["data"]["data"] = config.data.data

    return ckpt_dict


def data_log(logger, config):
    logger.log(
        f"[{config.data.data}] seed={config.seed} batch_size={config.data.batch_size}"
    )


def sde_log(logger, config_sde):
    sde_x = config_sde.x
    sde_adj = config_sde.adj
    logger.log(
        f"(X:{sde_x.type})=({sde_x.beta_min:.2f}, {sde_x.beta_max:.2f}) N={sde_x.num_scales} "
        f"(A:{sde_adj.type})=({sde_adj.beta_min:.2f}, {sde_adj.beta_max:.2f}) N={sde_adj.num_scales}"
    )


def model_log(logger, config):
    config_m = config.model
    model_log = (
        f"({config_m.model}): "
        f"depth={config_m.depth} nhid={config_m.nhid} "
        f"dropout={config_m.dropout}"
    )
    logger.log(model_log)


def start_log(logger, config, is_train=True):
    if is_train:
        logger.log("-" * 100)
        logger.log(f"{config.exp_name}")
    logger.log("-" * 100)
    data_log(logger, config)
    logger.log("-" * 100)


def train_log(logger, config):
    sde_log(logger, config.sde)
    model_log(logger, config)
    logger.log("-" * 100)


def sample_log(logger, configc):
    logger.log(f"[X] weight={configc.weight_x} [A] weight={configc.weight_adj}")
    logger.log("-" * 100)
