# AEV-PLIG

AEV-PLIG is a GNN-based scoring function that predicts the binding affinity of a bound protein-ligand complex given its 3D structure.

- [Installation guide](#installation-guide)
- [Demo](#demo)

## Installation guide
AEV-PLIG has been tested on the following systems:
+ macOS: Monterey (12.5.1)

### Create conda environment
Installation times may vary, but took around 30 seconds on Mac M1.
For *macOS*:
```
conda env create --file aev-plig-mac.yml
```
For *Linux*:
```
conda env create --file aev-plig-linux.yml
```
Install packages manually:
```
conda create --name aev-plig python=3.8
conda activate aev-plig
pip install torch torchvision torchaudio
pip install torch-scatter
pip install torch_geometric
pip install rdkit
pip install torchani
pip install qcelemental
pip install pandas
```

## Demo
This section demonstrates how to train your own AEV-PLIG model, and how to use AEV-PLIG to make predictions.

The computational requirements for each script are included, and unless otherwise specified, the hardware used is a Mac M1 CPU.

### Training

#### Download training data
Download the training datasets PDBbind and BindingNet
```
wget http://pdbbind.org.cn/download/PDBbind_v2020_other_PL.tar.gz
wget http://pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz
wget http://bindingnet.huanglab.org.cn/api/api/download/binding_database
```
Put PDBbind data into *data/pdbbind/refined-set* and *data/pdbbind/general-set*

Put BindingNet data into *data/bindingnet/from_chembl_client*

#### Generate PDBbind and BindingNet graphs
The following scripts will generate graphs into *pdbbind.pickle* and *bindingnet.pickle*. Takes around 30 minute in total to run.
```
python generate_pdbbind_graphs.py
python generate_bindingnet_graphs.py
```

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
