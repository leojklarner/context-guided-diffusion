import argparse


class Args_EDM(argparse.ArgumentParser):
    def __init__(
        self,
    ):
        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # data param
        self.add_argument("--dataset", default="cata", type=str)
        self.add_argument("--arg_id", type=int, help="arg_id")
        self.add_argument(
            "--split",
            type=str,
            default="",
            help="one of ['random_split', 'cluster_split']",
        )
        self.add_argument("--rings_graph", type=bool, default=True)
        self.add_argument("--max-nodes", default=11, type=str)

        # training param
        self.add_argument("--name", type=str, default="cata-test")
        self.add_argument("--restore", type=bool, default=None)
        self.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
        self.add_argument("--num_epochs", type=int, default=1000)
        self.add_argument("--normalize", type=bool, default=False)
        self.add_argument("--num-workers", type=int, default=32)
        self.add_argument("--batch-size", type=int, default=256)
        self.add_argument("--sample-rate", type=float, default=1)

        # Model parameters
        self.add_argument("--dp", type=eval, default=True, help="Data parallelism")
        self.add_argument("--clip_grad", type=eval, default=True, help="True | False")
        self.add_argument("--n_layers", type=int, default=9, help="number of layers")
        self.add_argument("--nf", type=int, default=192, help="number of layers")
        self.add_argument("--tanh", type=eval, default=True)
        self.add_argument("--attention", type=eval, default=True)
        self.add_argument("--coords_range", type=float, default=4)
        self.add_argument("--norm_constant", type=float, default=1)
        self.add_argument("--sin_embedding", type=eval, default=False)
        # EDM
        self.add_argument("--inv_sublayers", type=int, default=1)
        self.add_argument("--normalization_factor", type=float, default=1)
        self.add_argument("--aggregation_method", type=str, default="sum")
        self.add_argument("--diffusion_steps", type=int, default=1000)
        self.add_argument(
            "--diffusion_noise_schedule", type=str, default="polynomial_2"
        )
        self.add_argument(
            "--diffusion_noise_precision",
            type=float,
            default=1e-5,
        )
        self.add_argument("--diffusion_loss_type", type=str, default="l2")
        self.add_argument("--normalize_factors", type=eval, default=[3, 4, 10])

        # Logging
        self.add_argument("--save_dir", type=str, default="diffusion_training/")

        # unsused args for data loader compatibility
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
        self.add_argument(
            "--n_context_points",
            type=int,
            default=256,
            help="number of context points to sample per batch",
        )
        self.add_argument(
            "--context_set", type=str, default="all", help="one of ['all', 'rings_10']"
        )
