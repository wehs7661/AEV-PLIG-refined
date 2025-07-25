# AEV-PLIG refined
This is a repository that hosts a refined implementation for the AEV-PLIG model, a GNN-based scoring function that predicts the binding affinity of a bound protein-ligand complex given its 3D structure. Compared to the original repo ([here](https://github.com/isakvals/AEV-PLIG)), this repo provides user-friendly command-line interfaces (CLIs), a more organised codebase, and implements the method as a pip-installable package.

## Installation
We recommend using a conda environment to install the package.
```bash
conda create --name aev-plig
conda activate aev-plig
pip install --upgrade pip setuptools

git clone https://github.com/wehs7661/AEV-PLIG-refined.git
cd AEV-PLIG-refined
pip install -e .
```

Note that you might need to install PyTorch and PyTorch Geometric separately, depending on your system configuration.

```bash
pip install torch==2.5.0
python3.10 -c "import torch; print(torch.__version__)"
pip3.10 install torch-scatter --no-cache-dir -f https://data.pyg.org/whl/torch-<TORCH_VERSION>+<CUDA_VERSION>.html
```

For example, the following commands are for a system with CUDA 12.4 and PyTorch 2.5.0. Please adjust the commands according to your system's CUDA version and PyTorch version.

```bash
pip install torch==2.5.0
pip install torch-scatter --no-cache-dir -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

## Tutorial
In this tutorial, we will demonstrate how one can use the command-line interfaces (CLIs) provided in this package to conveniently train an ensemble of five AEV-PLIG models on the union set of HiQBind and BindingNet v1, then test the ensemble model on a custom test set. For details about reproducing results from the original paper, please refer to the original repo [here](https://github.com/isakvals/AEV-PLIG).

### Step 1: Prepare dataset index files using `process_dataset`

Before we generate a protein-ligand interaction graph (PLIG) for each binding complex, we need to generate an index file for each dataset of interest. Such an index file is required for each of the training sets (HiQBind, BindingNet v1, and a custom dataset) and the test set, and must at least contain the following columns:
- `system_id`: A unique identifier for each binding complex.
- `pK`: The binding affinity of the complex, which can be in pKd, pKi, or pIC50 format.
- `protein_path`: The path to the protein structure file (in PDB format).
- `ligand_path`: The path to the ligand structure file (in SDF format).
- `split`: The split of the binding complex, which can be `train`, `validation`, or `test`. This is used to indicate whether the complex is part of the training set, validation set, or test set.

For common datasets such as PDBBind, HiQBind, BindingDB, BindingNet v1, and BindingNet v2, such index files can be prepared using the `process_dataset` CLI. If you have a custom training set or test set, you will need to prepare a CSV file on your own and make sure it contains the necessary columns. Note, when using the `process_dataset` CLI, you can also provide a reference CSV file (using the `-cr` flag) to calculate the maximum Tanimoto similarity for each ligand in the processed dataset against the ligands in the reference CSV file, and only consider entries having a maximum similarity below the cutoff specified by the flag `-sc`. This is useful for ensuring that the ligands in your training set are not too similar to the ones in the reference dataset, which is often the test set to avoid data leakage. For more information about the CLI `process_dataset`, please refer to its help message, which can be accessed by running `process_dataset -h` or `process_dataset --help`.

In the following example, we will process the HiQBind and BindingNet v1 datasets using `process_dataset`, and assume that the python script `prepare_dataset.py` does the necessary processing to generate the index file for the custom dataset. Note that we set the similarity cutoff to 0.9 (with `-sc 0.9`), a split of 95% for training and 5% for validation (with `-s 95 5 0`), and a random seed of 0 (with `-rs 0`).
```bash
python prepare_dataset.py  # A custom script to prepare an index file for your custom test set, processed_custom_test.csv
process_dataset -ds hiqbind -d path_to_hiqbind_directory -cr path_to_ref_csv_file -l process_hiqbind.log -sc 0.9 -s 95 5 0 -rs 0 -o processed_hiqbind.csv
process_dataset -ds bindingnet_v1 -d path_to_bindingnet_v1_directory -cr path_to_ref_csv_file -l process_bindingnet_v1.log -sc 0.9 -s 95 5 0 -rs 0 -o processed_bindingnet_v1.csv
```
Note that log files are generated for each dataset processing step, which can be useful for debugging or tracking the processing steps. All the other CLIs in this package also generate informative log files.

### Step 2: Generate PLIGs using `generate_graphs`
After we have the index files ready, we use `generate_graphs` to generate PLIGs for each of the corresponding dataset, with the graphs saved in pickled files including `hiqbind.pickle`, `bindingnet_v1.pickle`, and `custom_test.pickle`. For more information about the CLI `generate_graphs`, please refer to its help message, which can be accessed by running `generate_graphs -h` or `generate_graphs --help`.

```bash
generate_graphs -c processed_hiqbind.csv -o hiqbind.pickle  -l generate_graphs_hiqbind.log
generate_graphs -c processed_bindinginet_v1.csv -o bindingnet_v1.pickle -l generate_graphs_bindingnet_v1.log
generate_graphs -c processed_custom_test.csv -o custom_test.pickle  -l generate_graphs_custom_test.log
```

### Step 3: Create PyTorch data using `create_pytorch_data`
After generating the PLIGs, we need to create PyTorch data files that can be used for training, validating and testing the AEV-PLIG models. This can be done using the `create_pytorch_data` CLI, which takes the pickled PLIGs and the corresponding index files as input, and generates PyTorch data files. For more information about the CLI `create_pytorch_data`, please refer to its help message, which can be accessed by running `create_pytorch_data -h` or `create_pytorch_data --help`.

```bash
create_pytorch_data -pg hiqbind.pickle bindingnet_v1.pickle custom_test_set.pickle -c processed_hiqbind.csv processed_bindingnet_v1.csv processed_test_set.csv -l create_pytorch_data.log
```
This command will generate a folder `data` that contains the subfolder `processed`, where the PyTorch files `dataset_test.pt`, `dataset_train.pt`, and `dataset_validation.pt` reside.

### Step 4: Train AEV-PLIG models using `train_aev_plig`
With the PyTorch data files ready, we can now train the AEV-PLIG models using the following command. In the following example, we train an ensemble of five AEV-PLIG models (with `-nm 5`). Check the help message of the CLI by running `train_aev_plig -h` or `train_aev_plig --help` if you are interested in trying different training parameters. Note that you can specify the GPU for training by setting the `CUDA_VISIBLE_DEVICES` environment variable, as done below.

```bash
CUDA_VISIBLE_DEVICES=0 train_aev_plig -nm 5 -l train_aev_plig.log
```
All trained models, which are named in the format `<date_and_time>_model_GATv2Net_dataset_*.model`, will be saved in the folder `outputs`. Additionally, a scaler file named `<date_and_time>_model_GATv2Net_dataset.pickle` will also be saved in the same folder.

### Step 5: Assess the trained models using `assess_aev_plig`
During the execution of the training command shown above, the performance of each trained model will be printed to the log file. If you would like to reassess the models after training, you can use the `assess_aev_plig` CLI. By default, this CLI assumes PyTorch files to be saved in folder `data/processed` and the trained models to be saved in the folder `outputs`, but you may specify different paths using the flags `-md`, `tr`, and `-t`. For more information about the CLI `assess_aev_plig`, please refer to its help message, which can be accessed by running `assess_aev_plig -h` or `assess_aev_plig --help`.

```bash
assss_trained_models -md outputs -tr data/processed/dataset_train.pt -v data/processed/dataset_validation.pt -t data/processed/dataset_test.pt -l assess_aev_plig.log -o assess_trained_models.csv
```
The command will output a CSV file `assess_trained_models.csv` that contains columns including `model`, `y_true`, `group_id`, `aboslute_error`, and `squared_error`. Performance metrics for each model and the ensemble model, such as RMSE, Pearson correaltion coefficient, Kendall's tau correlation coefficient, Spearman correlation coefficient, and C-index, will be printed to the log file.

### Step 6: Putting everything together in a shell script
To automate the above steps, we can create a shell script that runs all the necessary commands in sequence.

```bash
# Step 1. Prepare the dataset index files
python prepare_dataset.py  # A custom script you write to prepare your custom dataset index file
process_dataset -ds hiqbind -d path_to_hiqbind_directory -cr path_to_ref_csv_file -l process_hiqbind.log -sc 0.9 -s 95 5 0 -rs 0
process_dataset -ds bindingnet_v1 -d path_to_bindingnet_v1_directory -cr path_to_ref_csv_file -l process_bindingnet_v1.log -sc 0.9 -s 95 5 0 -rs 0

# Step 2. Generate PLIGs
generate_graphs -c processed_hiqbind.csv -o hiqbind.pickle  -l generate_graphs_hiqbind.log
generate_graphs -c processed_bindinginet_v1.csv -o bindingnet_v1.pickle -l generate_graphs_bindingnet_v1.log
generate_graphs -c processed_custom_test.csv -o custom_test.pickle  -l generate_graphs_custom_test.log

# Step 3. Create PyTorch data files
create_pytorch_data -pg hiqbind.pickle bindingnet_v1.pickle custom_test_set.pickle -c processed_hiqbind.csv processed_bindingnet_v1.csv processed_test_set.csv -l create_pytorch_data.log

# Step 4. Train AEV-PLIG models
CUDA_VISIBLE_DEVICES=0 train_aev_plig -nm 5 -l train_aev_plig.log

# Step 5. Assess the trained models
assss_trained_models -md outputs -tr data/processed/dataset_train.pt -v data/processed/dataset_validation.pt -t data/processed/dataset_test.pt -l assess_aev_plig.log -o assess_trained_models.csv
```

## References
If you use AEV-PLIG in your research, please cite the following paper:
```
@article{valsson2025narrowing,
  title={Narrowing the gap between machine learning scoring functions and free energy perturbation using augmented data},
  author={Valsson, {\'I}sak and Warren, Matthew T and Deane, Charlotte M and Magarkar, Aniket and Morris, Garrett M and Biggin, Philip C},
  journal={Communications Chemistry},
  volume={8},
  number={1},
  pages={41},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## Authors
Note that this project is based on [this repo](https://github.com/isakvals/AEV-PLIG). The authors of the changes made upon the original repo include:
- Wei-Tse Hsu, University of Oxford (wei-tse.hsu@bioch.ox.ac.uk)
- Savva Grevtsev, University of Oxford (savva.grevtsev@merton.ox.ac.uk)
