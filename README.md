# AEV-PLIG

AEV-PLIG is a GNN-based scoring function that predicts the binding affinity of a bound protein-ligand complex given its 3D structure.

## Installation
We recommend using a conda environment to install the package.
```
conda create --name aev-plig
git clone https://github.com/wehs7661/AEV-PLIG.git
cd AEV-PLIG
pip install .
```

## Reproducing the results from the paper
In this section, we elaborate necessary steps to reproduce the results from the paper.

### 1. Download training data
Execute the following commands to download and extract the training datasets PDBbind and BindingNet:
```
wget http://pdbbind.org.cn/download/PDBbind_v2020_other_PL.tar.gz
wget http://pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz
wget http://bindingnet.huanglab.org.cn/api/api/download/binding_database

tar -xvf PDBbind_v2020_other_PL.tar.gz
tar -xvf PDBbind_v2020_refined.tar.gz
mv binding_database binding_database.tar.gz
tar -xvf binding_database.tar.gz
```

### 2. Data processing
In the folder `AEV_PLIG_project`, we first create folders including `csv_files`, `log_files`, and `graphs` to organize the files generated during this step. 
```
mkdir csv_files log_files graphs
```
Notably, to make sure we have exactly the same training data as the original paper, we use the datasets `processed_pdbbind.csv` and `processed_bindingnet.csv` in the original repo as reference datasets.
```
cp {path_to_processed_pdbbind.csv} csv_files/ref_pdbbind.csv
cp {path_to_processed_bindingnet.csv} csv_files/ref_bindingnet.csv
```
Then, we use the CLI `process_dataset` to generate the preprocessed datasets using the following commands:
```
process_dataset -d {path_to_pdbbind} -ds pdbbind -r csv_files/ref_pdb
bind.csv -o csv_files/processed_pdbbind.csv  -l log_files/process_pdbbind.log
process_dataset -d {path_to_bindingnet_database} -ds bindingnet -r csv_files/ref_bindingnet.csv -o csv_files/processed_bindingnet.csv -l log_files/
process_bindingnet.log
```
This will generate a CSV file containing necessary columns for graph generation by the CLI `generate_graphs` (see below), including `system_id`, `protein_path`, and `ligand_path`. This step should take just a few seconds for PDBbind and under two minutes for BindingNet.

### 3. Graph generation
To generate graphs for PDBbind and BindingNet, we run the following command:
```
generate_graphs -c csv_files/processed_pdbbind.csv -o graphs/pdbbind_graphs.pickle -l log_files/generate_graphs_pdbbind.log
generate_graphs -c csv_files/processed_bindingnet.csv -o graphs/bindingnet_graphs.pickle -l log_files/generate_graphs_bindingnet.log
```
The command takes 30 to 60 minutes to complete. For each dataset, the generated graphs are saved in a pickle file.

---
#### Generate data for pytorch
Running this script takes around 2 minutes.
```
python create_pytorch_data.py
```
The script outputs the following files in *data/processed/*:

*pdbbind_U_bindingnet_ligsim90_train.pt*, *pdbbind_U_bindingnet_ligsim90_valid.pt*, and *pdbbind_U_bindingnet_ligsim90_test.pt*

#### Run training
Running the following script takes 25 hours using a NVIDIA GeForce GTX 1080 Ti
GPU. Once a model has been trained, the next section describes how to use it for predictions.
```
python training.py --activation_function=leaky_relu --batch_size=128 --dataset=pdbbind_U_bindingnet_ligsim90 --epochs=200 --head=3 --hidden_dim=256 --lr=0.00012291937615434127 --model=GATv2Net
```
The trained models are saved in *output/trained_models*


### Predictions
In order to make predictions, the model requires a *.csv* file with the following columns:
- *unique_id*, unique identifier for the datapoint
- *sdf_file*, relative path to the ligand *.sdf* file
- *pdb_file*, relative path to the protein *.pdb* file

An example dataset is included in *data/example_dataset.csv* for this demo.

```
python process_and_predict.py --dataset_csv=data/example_dataset.csv --data_name=example --trained_model_name=20240423-200034_model_GATv2Net_pdbbind_U_bindingnet_ligsim90
```
The script processes data in *dataset_csv*, and removes datapoints if:
1. .sdf file cannot be read by RDkit
2. Molecule contains rare element
3. Molecule has undefined bond type

The script then creates graphs and pytorch data to run the AEV-PLIG model specified with *trained_model_name*. The default is AEV-PLIG trained on PDBbind v2020 and BindingNet

The predictions are saved under *output/predictions/data_name_predictions.csv*

For the example dataset, the script takes around 20 seconds to run
