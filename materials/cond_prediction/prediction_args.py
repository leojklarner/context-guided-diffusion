import argparse
import pathlib


class PredictionArgs(argparse.ArgumentParser):
    def __init__(
        self,
    ):
        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # data param
        self.add_argument("--arg_id", type=int, help="arg_id")
        self.add_argument("--hyper_id", type=int, help="hyper_id")
        self.add_argument("--dataset", default="cata", type=str)
        self.add_argument(
            "--split",
            type=str,
            default="",
            help="one of ['random_split', 'cluster_split']",
        )
        self.add_argument(
            "--reg_type",
            type=str,
            default="",
            help="regression type, one of ['ps', 'fesb']",
        )
        self.add_argument(
            "--context_set", type=str, default="", help="one of ['all', 'rings_10']"
        )
        self.add_argument(
            "--n_context_points",
            type=int,
            default=256,
            help="number of context points to sample per batch",
        )
        self.add_argument("--rings_graph", type=bool, default=True)
        self.add_argument("--max-nodes", default=11, type=str)
        self.add_argument("--seed", default=0, type=int)
        # task param
        self.add_argument(
            "--target_features",
            # default="HOMO-LUMO gap/eV,HOMO,electron_affinity e/V,ionization e/V,"
            # "reorganisation energy eV,oxidation potential eV",
            default="LUMO_eV,GAP_eV,Erel_eV,aIP_eV,aEA_eV",
            type=str,
            help="list of the names of the target features in the csv file - can be multiple targets seperated with commas"
            "[HOMO_eV, LUMO_eV, GAP_eV, Dipmom_Debye, Etot_eV, Etot_pos_eV,"
            "Etot_neg_eV, aEA_eV, aIP_eV, Erel_eV]",
        )
        self.add_argument("--num_targets", type=int, default=5)
        # training param
        self.add_argument(
            "--name",
            type=str,
            default="cata-test",
        )
        self.add_argument("--restore", type=bool, default=None)
        self.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
        # self.add_argument("--weight_decay", type=float, default=1e-12, help="AdamW weight decay")
        self.add_argument("--num_epochs", type=int, default=200)
        self.add_argument("--early_stopping_patience", type=int, default=5)
        self.add_argument("--normalize", type=bool, default=False)
        self.add_argument("--batch-size", type=int, default=256)
        self.add_argument("--sample-rate", type=float, default=1.0)
        self.add_argument("--num-workers", type=int, default=0)

        # Model parameters
        self.add_argument("--dp", type=eval, default=True, help="Data parallelism")
        self.add_argument("--n_layers", type=int, default=12, help="number of layers")
        self.add_argument(
            "--nf", type=int, default=196, help="number of hidden dims"
        )  # was originally 196
        self.add_argument("--tanh", type=eval, default=True)
        self.add_argument("--attention", type=eval, default=True)
        self.add_argument("--coords_range", type=float, default=4)
        self.add_argument("--norm_constant", type=float, default=1)
        self.add_argument("--normalization_factor", type=float, default=1)

        # Logging
        self.add_argument("--save_dir", type=str, default="standard_regression/")

        # FS-EB args
        self.add_argument("--prior_likelihood_cov_scale", type=float, default=1e0)
        self.add_argument("--prior_likelihood_cov_diag", type=float, default=1e2)
        self.add_argument("--prior_var", type=float, default=1e3)
        self.add_argument(
            "--log_prior_likelihood_var",
            type=float,
            default=2,
            help="Target variance of FS-EB prior",
        )
        self.add_argument(
            "--log_likelihood_var_clip_min",
            type=float,
            default=-4,
            help="Lower bound to clip predicted variance to",
        )
