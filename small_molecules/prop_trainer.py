import os
import time
from tqdm import tqdm
import numpy as np
import torch

from utils.loader import (
    load_seed,
    load_device,
    load_sde,
    load_prop_data,
    load_context_data,
)
from utils.logger import Logger, set_log
from utils.regressor_utils import (
    load_regressor_params,
    load_regressor_batch,
    load_regressor_optimizer,
    load_regressor_loss_fn,
    start_log,
    train_log,
    load_regressor,
    get_context_logit_mean,
)

torch.autograd.set_detect_anomaly(True)

class DANNModel(torch.nn.Module):
    """
    One linear layer for domain classification.
    """

    def __init__(self, in_features, out_features):
        super(DANNModel, self).__init__()
        self.layer = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer(x)

    def loss(self, x, labels):
        preds = self.forward(x)
        return torch.nn.functional.cross_entropy(preds, labels)


class Trainer(object):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(
            self.config, f"{self.config.train.reg_type}{'_c' + str(self.config.data.context_size) if self.config.train.reg_type == 'fseb' else ''}"
        )

        self.device = load_device(0)

        self.train_loader, self.test_loader = load_prop_data(self.config, self.device)

        if self.config.train.reg_type in ["fseb", "coral", "dann", "domain_confusion"]:
            self.context_loader = load_context_data(self.config, self.device)
        else:
            self.context_loader = None

        self.params = load_regressor_params(self.config)
        if "ensemble" in self.config.train.reg_type:
            self.params["ensemble_size"] = self.config.train.ensemble_size

        load_seed(self.config.seed)

    def train(self):
        self.config.exp_name = time.strftime("%b%d-%H:%M:%S", time.gmtime())
        os.makedirs(
            f"./checkpoints/{self.config.data.data}/{self.config.train.reg_type}{'_c' + str(self.config.data.context_size) if self.config.train.reg_type == 'fseb' else ''}",
            exist_ok=True,
        )
        self.ckpt = f"{self.config.train.reg_type}{'_c' + str(self.config.data.context_size) if self.config.train.reg_type == 'fseb' else ''}/prop_{self.config.train.prop}"
        print("\033[91m" + f"{self.ckpt}" + "\033[0m")

        self.model, self.optimizer, self.scheduler = load_regressor_optimizer(
            self.params, self.config.train, self.device
        )

        assert self.config.train.reg_type in [
            "fseb",
            "ps",
            "weight_decay",
            "weight_decay_ensemble",
            "weight_decay_pretrained",
            "coral",
            "dann",
            "domain_confusion",
        ], f"Invalid regularization type: {self.config.train.reg_type}"

        if self.config.train.reg_type == "fseb":
            print("Using FSEB regulariser and context model")
            self.context_model = load_regressor(params=self.params).to(
                f"cuda:{self.device[0]}"
            )
            self.context_model.eval()

            # get mean of label distribution of training data to scale the context model logits
            train_y_mean = self.train_loader.dataset.tensors[-1].mean()
            print("\n\ntrain_y_mean: ", train_y_mean)

            context_logits_mean = get_context_logit_mean(
                self.config, self.context_model, self.context_loader
            )

            print("context_logits_mean: ", context_logits_mean)

        elif self.config.train.reg_type == "coral":
            print("Using CORAL regularizer")
            self.context_model = 5
            train_y_mean = None
            context_logits_mean = None

        elif self.config.train.reg_type == "dann" or self.config.train.reg_type == "domain_confusion":
            print("Using DANN regularizer")
            self.context_model = DANNModel(
                in_features=self.config.model.nhid, out_features=2
            ).to(f"cuda:{self.device[0]}")
            train_y_mean = None
            context_logits_mean = None

        else:
            self.context_model = None
            print("Not using context model")
            train_y_mean = None
            context_logits_mean = None

        self.eps = self.config.train.eps
        self.sde_x = load_sde(self.config.sde.x)
        self.sde_adj = load_sde(self.config.sde.adj)

        logger = Logger(os.path.join(self.log_dir, f"{self.ckpt}.log"), mode="a")
        os.makedirs(
            os.path.join(self.log_dir, f"{self.config.train.reg_type}{'_c' + str(self.config.data.context_size) if self.config.train.reg_type == 'fseb' else ''}"), exist_ok=True
        )
        start_log(logger, self.config)
        train_log(logger, self.config)

        logger.log(str(self.model))
        logger.log("-" * 100)

        self.loss_fn = load_regressor_loss_fn(config=self.config, device=self.device)
        epoch_times = []

        for epoch in range(self.config.train.num_epochs):
            self.train_loss = []
            self.train_fseb_reg = []
            self.train_ps_reg = []
            self.test_loss = []
            self.train_corr = []
            self.test_corr = []
            t_start = time.time()

            self.model.train()
            for i, train_b in enumerate(
                tqdm(self.train_loader, desc=f"[Epoch {epoch+1}]")
            ):
                x, adj, labels = load_regressor_batch(train_b)

                # get context points
                if self.config.train.reg_type in ["fseb", "coral", "dann", "domain_confusion"]:
                    x_context, adj_context, _ = load_regressor_batch(
                        next(iter(self.context_loader))
                    )
                else:
                    x_context, adj_context = None, None

                self.optimizer.zero_grad()

                loss, fseb_reg, ps_reg, corr = self.loss_fn(
                    model=self.model,
                    x=x,
                    adj=adj,
                    labels=labels,
                    context_model=self.context_model,
                    x_context=x_context,
                    adj_context=adj_context,
                    reg_type=self.config.train.reg_type,
                    train_y_mean=train_y_mean,
                    context_logits_mean=context_logits_mean,
                )

                # add regularization terms

                reg_scale = self.config.data.batch_size * len(self.train_loader)
                # reg_scale = self.config.data.context_size * len(self.train_loader)
                fseb_reg = fseb_reg / reg_scale
                fseb_reg = fseb_reg * (512 / self.config.data.context_size)
                ps_reg = ps_reg / reg_scale

                if isinstance(loss, list):
                    for l in loss:
                        combined_loss = l + fseb_reg + ps_reg
                        combined_loss.backward()
                    loss = np.mean([l.detach().cpu().numpy() for l in loss])

                else:
                    combined_loss = loss + fseb_reg + ps_reg
                    combined_loss.backward()
                
                self.optimizer.step()

                self.train_loss.append(loss.item())
                self.train_fseb_reg.append(fseb_reg.item())
                self.train_ps_reg.append(ps_reg.item())
                self.train_corr.append(corr)

            if self.config.train.lr_schedule:
                self.scheduler.step()

            epoch_times.append(time.time() - t_start)

            logger.log(
                f"Epoch: {epoch+1:03d} | {time.time()-t_start:.2f}s | "
                f"TRAIN loss: {np.mean(self.train_loss):.4e} | "
                f"TRAIN fseb_reg: {np.mean(self.train_fseb_reg):.4e} | "
                f"TRAIN ps_reg: {np.mean(self.train_ps_reg):.4e} | "
                f"TRAIN corr: {np.mean(self.train_corr):.4f} | ",
                verbose=False,
            )

        self.model.eval()
        for _, test_b in enumerate(self.test_loader):
            x, adj, labels = load_regressor_batch(test_b)

            with torch.no_grad():
                loss, fseb_reg, ps_reg, corr = self.loss_fn(
                    model=self.model,
                    x=x,
                    adj=adj,
                    labels=labels,
                    context_model=None,
                    x_context=None,
                    adj_context=None,
                    reg_type=self.config.train.reg_type,
                    train_y_mean=None,
                    context_logits_mean=None,
                )
                if isinstance(loss, list):
                    for l in loss:
                        self.test_loss.append(l.item())
                else:
                    self.test_loss.append(loss.item())

                self.test_corr.append(corr)

        logger.log(
            f"Epoch: {epoch+1:03d} | {time.time()-t_start:.2f}s | "
            f"TRAIN loss: {np.mean(self.train_loss):.4e} | "
            f"TRAIN fseb_reg: {np.mean(self.train_fseb_reg):.4e} | "
            f"TRAIN ps_reg: {np.mean(self.train_ps_reg):.4e} | "
            f"TRAIN corr: {np.mean(self.train_corr):.4f} | "
            f"TEST loss: {np.mean(self.test_loss):.4e} | "
            f"TEST corr: {np.mean(self.test_corr):.4f}",
            verbose=False,
        )

        if self.config.train.reg_type == "fseb":
            logger.log(
                f"prior_likelihood_cov_diag: {self.config.train.prior_likelihood_cov_diag}\n"
                f"prior_likelihood_cov_scale: {self.config.train.prior_likelihood_cov_scale}\n"
                f"context_size: {self.config.data.context_size}\n"
            )

        checkpoint_path = f"./checkpoints/{self.config.data.data}/{self.ckpt}.pth"

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            if np.mean(self.test_loss) < checkpoint["test_loss"]:

                print("Previous checkpoint loss: ", checkpoint["test_loss"])
                print("Current checkpoint loss: ", np.mean(self.test_loss))

                torch.save(
                    {
                        "model_config": self.config,
                        "params": self.params,
                        "test_loss": np.mean(self.test_loss),
                        "test_corr": np.mean(self.test_corr),
                        "state_dict": self.model.state_dict(),
                        "average_batch_time": np.mean(epoch_times) / len(self.train_loader),
                    },
                    checkpoint_path,
                )
        else:

            torch.save(
                {
                    "model_config": self.config,
                    "params": self.params,
                    "test_loss": np.mean(self.test_loss),
                    "test_corr": np.mean(self.test_corr),
                    "state_dict": self.model.state_dict(),
                    "average_batch_time": np.mean(epoch_times) / len(self.train_loader),
                },
                checkpoint_path,
            )
