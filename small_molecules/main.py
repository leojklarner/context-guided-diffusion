import os
import argparse
import torch
from copy import deepcopy
from parsers.parser import Parser
from parsers.config import get_config

from cProfile import Profile
from pstats import SortKey, Stats

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def main(work_type_args):
    args = Parser().parse()
    config = get_config(args.config, args.gpu, args.seed)

    if work_type_args.type == 'train':

        print("\n\n\nTraining config:")
        print(config)

        from prop_trainer import Trainer
        Trainer(config).train()

        # with Profile() as profile:
        #     Trainer(config).train()
        #     Stats(profile).strip_dirs().sort_stats('cumtime').print_stats()

    elif work_type_args.type == 'sample':

        print("\n\n\nSampler config:")
        print(config)

        from sampler import Sampler

        for i in range(5):

            sample_config = deepcopy(config)
            sample_config.seed = config.seed + i

            print(f"\n\n\nHyperparameters of sampling run {i}:")
            print(sample_config)

            result_dir = os.path.join("generated_samples", sample_config.model.prop.ckpt)
            os.makedirs(result_dir, exist_ok=True)

            results = Sampler(sample_config).sample()
            results.to_pickle(os.path.join(result_dir, f"samples_{sample_config.model.prop.weight_x}_{i}.pkl"))

    elif work_type_args.type == 'retrain_best':

        from prop_trainer import Trainer
        from sampler import Sampler

        print("\n\n\nSampler config:")
        print(config)

        # load best model
        best_train_config = torch.load(f"checkpoints/ZINC250k/{config.model.prop.ckpt}.pth")['model_config']

        print("\n\n\nOriginal best hyperparameters:")
        print(best_train_config)

        for i in range(5):

            train_config = deepcopy(best_train_config)
            train_config.train.prop = best_train_config.train.prop + f'_retrain_{i}'
            train_config.seed = best_train_config.seed + i

            print(f"\n\n\nHyperparameters of retraining run {i}:")
            print(train_config)

            if os.path.exists(f"checkpoints/ZINC250k/{config.model.prop.ckpt}_retrain_{i}.pth"):
                print(f"\n\n\nCheckpoint checkpoints/ZINC250k/{config.model.prop.ckpt}_retrain_{i}.pth already exists, skipping retrain run...")
            else:
                print(f"\n\n\nCheckpoint checkpoints/ZINC250k/{config.model.prop.ckpt}_retrain_{i}.pth does not exist, starting retrain training...")
                Trainer(train_config).train()

            sample_config = deepcopy(config)
            sample_config.seed = train_config.seed
            sample_config.model.prop.ckpt = config.model.prop.ckpt + f'_retrain_{i}'

            print(f"\n\n\nHyperparameters of sampling run {i}:")
            print(sample_config)

            result_dir = os.path.join("generated_samples_retrain", sample_config.model.prop.ckpt)
            os.makedirs(result_dir, exist_ok=True)

            results = Sampler(sample_config).sample()
            results.to_pickle(os.path.join(result_dir, f"samples_{sample_config.model.prop.weight_x}_retrain_{i}.pkl"))

    else:
        raise ValueError(f'Wrong type {work_type_args.type}')


if __name__ == '__main__':
    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('-t', '--type', type=str, required=True)

    main(work_type_parser.parse_known_args()[0])
