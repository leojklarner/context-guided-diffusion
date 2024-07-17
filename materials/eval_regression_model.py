import os
import json
import random
import shutil
from datetime import datetime
from time import time, sleep
import warnings

from torch.nn.functional import l1_loss, mse_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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

import numpy as np
import torch

from torch import optim, Tensor, linspace


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if variable.shape[-1] != 0:
            assert_correctly_masked(variable, node_mask)


def normalize(x, h, node_mask):
    normalize_factors = [3, 4, 10]
    norm_biases = (None, 0.0, 0.0)

    x = x / normalize_factors[0]

    # Casting to float in case h still has long or int type.
    h_cat = (
        (h["categorical"].float() - norm_biases[1]) / normalize_factors[1] * node_mask
    )
    h_int = (h["integer"].float() - norm_biases[2]) / normalize_factors[2]

    # Create new h dictionary.
    h = {"categorical": h_cat, "integer": h_int}

    return x, h


def compute_loss(
    model,
    x,
    h,
    node_mask,
    edge_mask,
    target,
) -> Tensor:
    # add normalization before passing to model?

    x, h = normalize(
        x,
        {"categorical": h, "integer": torch.zeros(0).to(x.device)},
        node_mask,
    )

    xh = torch.cat([x, h["categorical"]], dim=-1)

    bs, n_nodes, n_dims = x.size()
    assert_correctly_masked(x, node_mask)
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    pred = model(xh, node_mask, edge_mask)
    loss = mse_loss(pred, target)
    return loss, (pred - target).abs().detach()


def train_epoch(
    epoch,
    cond_predictor,
    dataloader,
    optimizer,
    args,
    writer,
):
    cond_predictor.train()

    start_time = time()
    loss_list = []
    rl_loss = []
    with tqdm(dataloader, unit="batch", desc=f"Train {epoch}") as tepoch:
        for i, (x, node_mask, edge_mask, node_features, y) in enumerate(tepoch):
            x = x.to(args.device)
            y = y.to(args.device)
            node_mask = node_mask.to(args.device).unsqueeze(2)
            edge_mask = edge_mask.to(args.device)
            h = node_features.to(args.device)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            loss, _ = compute_loss(cond_predictor, x, h, node_mask, edge_mask, y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            rl_loss.append(dataloader.dataset.rescale_loss(loss).item())

            tepoch.set_postfix(loss=np.mean(loss_list).item())
    print(
        f"[{epoch}|train] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
        f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
        f" in {int(time()-start_time)} secs"
    )
    sleep(0.01)
    writer.add_scalar("Train loss", np.mean(loss_list), epoch)
    writer.add_scalar("Train L1 (rescaled)", np.mean(rl_loss), epoch)


def val_epoch(
    tag,
    epoch,
    cond_predictor,
    dataloader,
    args,
    writer,
):
    cond_predictor.eval()
    with torch.no_grad():
        start_time = time()
        loss_list = []
        rl_loss = []
        # with tqdm(dataloader, unit="batch", desc=f"{tag} {epoch}") as tepoch:
        for i, (x, node_mask, edge_mask, node_features, y) in enumerate(dataloader):
            x = x.to(args.device)
            y = y.to(args.device)
            node_mask = node_mask.to(args.device).unsqueeze(2)
            edge_mask = edge_mask.to(args.device)
            h = node_features.to(args.device)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            loss, _ = compute_loss(
                cond_predictor,
                x,
                h,
                node_mask,
                edge_mask,
                y,
            )

            loss_list.append(loss.item())
            rl_loss.append(dataloader.dataset.rescale_loss(loss).item())

            # tepoch.set_postfix(loss=np.mean(loss_list).item())
        print(
            f"[{epoch}|{tag}] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
            f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
            f" in {int(time() - start_time)} secs"
        )
        sleep(0.01)
        writer.add_scalar(f"{tag} loss", np.mean(loss_list), epoch)
        writer.add_scalar(f"{tag} L1 (rescaled)", np.mean(rl_loss), epoch)

    return np.mean(loss_list)


def get_cond_predictor_model(args, dataset: AromaticDataset):
    cond_predictor = EGNN_predictor(
        in_nf=dataset.num_node_features,
        device=args.device,
        hidden_nf=args.nf,
        out_nf=dataset.num_targets,
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
    return cond_predictor


def main(pred_args):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(pred_args)

    print("\n\nTrain size:", len(train_loader.dataset))
    print("Val size:", len(val_loader.dataset))
    print("Test size:", len(test_loader.dataset))

    cond_predictor = get_cond_predictor_model(pred_args, train_loader.dataset)
    optimizer = optim.AdamW(
        cond_predictor.parameters(),
        lr=pred_args.lr,
        amsgrad=True,
        weight_decay=pred_args.weight_decay,
    )

    # Save path
    writer = SummaryWriter(log_dir=pred_args.exp_dir)

    # Run training
    print("Begin training")
    best_val_loss = 1e9
    best_epoch = 0
    for epoch in range(pred_args.num_epochs):
        train_epoch(
            epoch,
            cond_predictor,
            train_loader,
            optimizer,
            pred_args,
            writer,
        )
        val_loss = val_epoch(
            "val",
            epoch,
            cond_predictor,
            val_loader,
            pred_args,
            writer,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(cond_predictor.state_dict(), pred_args.exp_dir + "/model.pt")

    print(f"{best_epoch=}, {best_val_loss=:.4f}")
    cond_predictor.load_state_dict(torch.load(pred_args.exp_dir + "/model.pt"))
    print("Test all times:")
    val_epoch(
        "test",
        epoch,
        cond_predictor,
        test_loader,
        pred_args,
        writer,
    )
    writer.close()


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    pred_args = PredictionArgs().parse_args()

    split_map = {0: "random_split", 1: "cluster_split"}
    pred_args.name = f"{pred_args.name}_{split_map[pred_args.split]}"

    pred_args.exp_dir = f"{pred_args.save_dir}/{pred_args.name}"
    print(pred_args.exp_dir)

    # Create model directory
    if not os.path.isdir(pred_args.exp_dir):
        os.makedirs(pred_args.exp_dir)

    with open(pred_args.exp_dir + "/args.txt", "w") as f:
        json.dump(pred_args.__dict__, f, indent=2)
    # Automatically choose GPU if available
    pred_args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("CUDA:", torch.cuda.is_available())
    print("Pred args:", pred_args)

    # Where the magic is
    main(pred_args)
