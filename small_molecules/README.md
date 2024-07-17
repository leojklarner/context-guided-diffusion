# Section 5.1: Graph-Structured Diffusion Models For Small Molecules

This directory contains the code for the experiments with graph-structured diffusion models applied to the generation of potent and synthesizable drug-like small molecules, as described in Section 5.1 of the paper. The code is based on the work of Lee et al., ICML 2023, which is available at https://github.com/SeulLee05/MOOD. The original README is included below.

The following instructions provide a step-by-step guide to running the code and reproducing the experiments and results in the paper.

### 1. Environment Setup

The code uses the Python 3.8 environment provided by Lee et al. and a handful of additional packages. To install the required dependencies, run the following command:

```
conda env create -n <env_name> --file environment.yml
conda activate <env_name>
```

### 2. Data Preparation

Both the labeled training dataset (`zinc_250k.csv`) and the unlabelled context data (`zinc_500k.csv`) that were used for our experiments are available in the `data` directory. The molecules in the labeled `zinc_250k.csv` set are assigned to the protein-specific low-property training or high-property validation sets based on the assignments in the `valid_idx_zinc250k_<protein_name>.json` files that are generated in the `data/data_exploration.ipynb` notebook. Before running the experiments, the data needs to be preprocessed. To do so, run the following command:

```
python data/preprocess.py --dataset <dataset_name>
```

where a `<dataset_name>` of `ZINC250k` refers to the training set and `ZINC500k` refers to the context set. A utility SLURM script to process both datasets is provided in `process_data.sh`.

### 3. Preparing config files for hyperparameter tuning

The hyperparameters for all experiments are defined in the configuration files in the `config` directory. Following the setup in the original codebase, a separate `.yaml` file is created for each hyperparameter combination with the `yaml_factory.ipynb` notebooks in the respective subdirectories. The config files are named as follows:

```
prop_train_<reg_type>/prop_train_<protein_target>_<hyper_id>
```

where `<reg_type>` is the type of regularization used in the training of the property predictor, `<protein_target>` specifies which of the five protein targets (encoded as {0, 1, 2, 3, 4}) to use, and `<hyper_id>` is an integer id of the specific hyperparameter combination that is used. Please run all cells in the appropriate notebooks before starting the experiments.

### 4. Training the guidance models

To train the guidance models, run the following command:

```
python main.py --type train --config <config_file>
``` 

with the config file defined as above. The training process will save the model checkpoints with the best performance on the held-out high property validation set to the `checkpoints/` directory. Utility SLURM scripts to train the different models are provided in `train_regressor_ours.sh`, `train_regressor_ps.sh`, `train_regressor_weight_decay.sh`, `train_ensemble_weight_decay.sh`, and `train_pretrained_weight_decay.sh`, respectively. 

### 5. Generating molecules

To generate molecules using the trained guidance models and pre-trained graph-structured diffusion model, run the following command:

``` 
python -u main.py --type retrain_best --config <config_file>
```

with the config file defined as above. This command loads the optimal hyperparameter combination and automatically retrains 5 independent models with different random seeds. It then generates 500 samples each and saves the results in the `generated_samples_retrain` directory. 

### 6. Evaluating the generated molecules

Once the molecules have been generated, the `analyse_results.ipynb` notebook can be used to evaluate the performance of each guidance model. This notebook loads and compares the properties of the generated samples and summarises them as plots.

---

<h1 align="center">Exploring Chemical Space with<br>Score-based Out-of-distribution Generation</h1>

This is the official code repository for the paper [Exploring Chemical Space with Score-based Out-of-distribution Generation](https://arxiv.org/abs/2206.07632) (ICML 2023), in which we propose *Molecular Out-Of-distribution Diffusion (MOOD)*.

<p align="center">
    <img width="750" src="assets/concept.png"/>
</p>

## Contribution

+ We propose a novel score-based generative model for OOD generation, which *overcomes the limited explorability of previous models* by leveraging the novel OOD-controlled reverse-time diffusion.
+ Since the extended exploration space by the OOD control contains molecules that are chemically implausible, we propose a framework for molecule optimization that *leverages the gradients of the property prediction network* to confine the generated molecules to a novel yet chemically meaningful space.
+ We experimentally demonstrate that the proposed MOOD can generate *novel molecules that are drug-like, synthesizable, and have high binding affinity* for five protein targets, outperforming existing molecule generation methods.

## Dependencies
Run the following commands to install the dependencies:
```bash
conda create -n mood python=3.8
conda activate mood
conda install -c pytorch pytorch==1.12.0 cudatoolkit=11.3
conda install -c conda-forge rdkit=2020.09 openbabel
pip install tqdm pyyaml pandas easydict networkx==2.6.3 numpy==1.20.3
chmod u+x scorer/qvina02
```

## Running Experiments

### 1. Preparation
MOOD utilizes [GDSS](https://github.com/harryjo97/GDSS) as its backbone diffusion model. In our paper, we utilized the pretrained `gdss_zinc250k_v2.pth` GDSS checkpoint, which is in the folder `checkpoints/ZINC250k`.

Run the following command to preprocess the ZINC250k dataset:
```bash
python data/preprocess.py
```

### 2. Training a Property Prediction Network $P_\phi$
We provide the pretrained property predictor networks ($P_\text{obj}=\hat{\text{DS}} \times \text{QED} \times \hat{\text{SA}}$) for target proteins parp1, fa7, 5ht1b, braf, and jak2, respectively (`prop_parp1.pth`, `prop_fa7.pth`, `prop_5ht1b.pth`, `prop_braf.pth`, and `prop_jak2.pth`, respectively), in the folder `checkpoints/ZINC250k`.

To train your own property predictor, run the following command:
```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python main.py --type train --config prop_train
```
You can modify hyperparameters in `config/prop_train.yaml`.

### 3. Generation and Evaluation
To generate molecules, run the following command:
```sh
CUDA_VISIBLE_DEVICES=${gpu_id} python main.py --type sample --config sample
```
You can modify hyperparameters in `config/sample.yaml`.

## Citation
If you find this repository and our paper useful, we kindly request to cite our work.

```BibTex
@article{lee2023MOOD,
  author    = {Seul Lee and Jaehyeong Jo and Sung Ju Hwang},
  title     = {Exploring Chemical Space with Score-based Out-of-distribution Generation},
  journal   = {Proceedings of the 40th International Conference on Machine Learning},
  year      = {2023}
}
```
