import os
import time
import torch
import numpy as np
import pandas as pd

from utils.logger import Logger, set_log, start_log, train_log, check_log
from utils.loader import load_ckpt, load_model_from_ckpt, load_data, \
                         load_seed, load_device, load_sampling_fn
from utils.graph_utils import init_flags, quantize_mol
from utils.mol_utils import gen_mol, mols_to_smiles

from utils.regressor_utils import load_regressor_from_ckpt, load_regressor_ckpt
from utils.regressor_utils import train_log as train_log_prop
from utils.regressor_utils import sample_log as sample_log_prop

from evaluate import evaluate


class Sampler(object):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.device = load_device(self.config.gpu)
    
    def sample(self):
        self.diff = load_ckpt(self.config, self.device)['diff']
        self.config_diff = self.diff['config']

        self.prop = load_regressor_ckpt(self.config, self.device)['prop']
        self.config_prop = self.prop['config']

        print("\nUsing property prediction config:\n")
        print(self.config_prop)
        print("\nAnd diffusion config:\n")
        print(self.config_diff)

        self.check_config(self.config_diff, self.config_prop)
        load_seed(self.config.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.config_diff, foldername=self.config_prop.train.reg_type, is_train=False)
        self.log_name = f"{self.config.model.prop.ckpt}_{self.config.seed}"

        os.makedirs(os.path.join(self.log_dir, f"{self.config_prop.train.reg_type}{'_c' + str(self.config_prop.data.context_size) if self.config_prop.train.reg_type == 'fseb' else ''}"), exist_ok=True)
        print(f'logname: {self.log_name}')
        print(f'logdir: {self.log_dir}')
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            start_log(logger, self.config_diff)
            train_log(logger, self.config_diff)
            train_log_prop(logger, self.config_prop)
            sample_log_prop(logger, self.config.model.prop)
        
        logger.log(f'snr={self.config.model.diff.snr} seps={self.config.model.diff.scale_eps} n_steps={self.config.model.diff.n_steps}')

        self.model_x = load_model_from_ckpt(self.diff['params_x'], self.diff['x_state_dict'], self.device)
        self.model_adj = load_model_from_ckpt(self.diff['params_adj'], self.diff['adj_state_dict'], self.device)

        self.regressor = load_regressor_from_ckpt(self.prop['params'], self.prop['state_dict'], self.device)
        self.regressor.eval()
        self.sampling_fn = load_sampling_fn(self.config_diff, self.config.model.diff, self.config.model.prop,
                                            self.config.sample, self.device)

        t_start = time.time()

        self.train_graph_list, _ = load_data(self.config_prop, get_graph_list=True)     # for init_flags

        self.init_flags = init_flags(self.train_graph_list, self.config_diff, self.config.sample.num_samples).to(self.device[0])

        x, adj = self.sampling_fn(self.model_x, self.model_adj, self.init_flags, self.regressor)

        logger.log(f"{time.time()-t_start:.2f} sec elapsed for sampling")


        print("samples_int dtype", torch.unique(adj), adj.dtype)
        
        samples_int = quantize_mol(adj)

        print("samples_int unique", np.unique(samples_int))
        print("samples_int dtype", samples_int.dtype)
        print("samples int nan ratio", np.isnan(samples_int).mean())

        samples_int_ = samples_int - 1
        samples_int_[samples_int_ == -1] = 3

        print("samples_int unique", np.unique(samples_int_))
        print("samples_int dtype", samples_int_.dtype)
        print("samples int nan ratio", np.isnan(samples_int_).mean())

        adj = torch.nn.functional.one_hot(torch.tensor(samples_int_), num_classes=4).permute(0,3,1,2)
        x = torch.where(x > 0.5, 1, 0)
        x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)

        gen_mols, _ = gen_mol(x, adj, self.config_diff.data.data)
        gen_smiles = mols_to_smiles(gen_mols)

        if 'parp1' in self.config_prop.train.prop: protein = 'parp1'
        elif 'fa7' in self.config_prop.train.prop: protein = 'fa7'
        elif '5ht1b' in self.config_prop.train.prop: protein = '5ht1b'
        elif 'braf' in self.config_prop.train.prop: protein = 'braf'
        elif 'jak2' in self.config_prop.train.prop: protein = 'jak2'
        else: raise NotImplementedError(f'protein {self.config_prop.train.prop} not implemented')

        tmp_dir = f"{self.config.model.prop.ckpt}/{self.config.seed}"

        result = evaluate(protein, tmp_dir, os.path.join(self.log_dir, self.log_name), gen_smiles, gen_mols)
        
        logger.log(f'Validity: {result["validity"]}')
        logger.log(f'Uniqueness: {result["uniqueness"]}')
        logger.log(f'Novelty (sim. < 0.4): {result["novelty"]}')
        logger.log(f'Novel top 5% DS (QED > 0.5, SA < 5, sim. < 0.4): '
                f'{result["top_ds"]} ± {result["top_ds"]:.4f}')
        logger.log(f'Novel hit ratio (QED > 0.5, SA < 5, sim. < 0.4): {result["hit"] * 100:.4f} %')

        logger.log(f"{time.time()-t_start:.2f} sec elapsed for docking simulation")

        logger.log('='*100)

        #print(result)

        return pd.DataFrame(result)
        
    def check_config(self, config1, config2):
        assert config1.data.batch_size == config2.data.batch_size, 'Batch size Mismatch'
        assert config1.data.max_node_num == config2.data.max_node_num, 'Max node num Mismatch'
        assert config1.data.max_feat_num == config2.data.max_feat_num, 'Max feat. num Mismatch'
        assert config1.sde.x == config2.sde.x, 'SDE Mismatch: X'
        assert config1.sde.adj == config2.sde.adj, 'SDE Mismatch: Adj'
