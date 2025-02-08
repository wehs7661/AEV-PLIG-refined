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

## Demo
This section demonstrates how to train your own AEV-PLIG model, and how to use AEV-PLIG to make predictions.

### Training

#### Download training data
Download the training datasets PDBbind and BindingNet using the following commands:
```
wget http://pdbbind.org.cn/download/PDBbind_v2020_other_PL.tar.gz
wget http://pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz
wget http://bindingnet.huanglab.org.cn/api/api/download/binding_database
```

#### Preprocess training data
Before generating graphs for a dataset, we need to preprocess the dataset. Taking PDBbind as an example, we run the following command:
```
process_dataset -ds pdbbind -d {path_to_pdbbind_dataset} -o {output_directory} 
```
This will generate a CSV file containing necessary columns for graph generation by the CLI `generate_graphs` (see below), including `system_id`, `protein_path`, and `ligand_path`. This should take just a few seconds.

#### Generate PDBbind and BindingNet graphs
To generate graphs for a dataset, e.g., PDBbind, run the following command:
```
generate_graphs -c processed_pdbbind.csv -o graphs_pdbbind.pickle
```
The command takes around 30 minutes in total to complete. The generated graphs are saved in a pickle file.


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
