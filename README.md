# Introduction
This repository contains the PyTorch implementation of **PepBAN** framework, predicting peptide-protein interactions based on sequences and deep learning.

# Installion Guide
## create a new conda environment
$ conda create --name PepBAN python=3.8.13
$ conda activate PepBAN 
$ pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

## install according to the requirements
$ pip install -r requirements.txt

# Datasets
The `datasets` folder contains all experimental data used in PepBAN: BioGRID, STRING, camp, hsm and cyclic peptide.

# Configs
The `configs` folder provides optim hyperparameters and the optim hyperparameters are designed by "hy.py" through the command:
```
python hy.py
```
The optimized parameter file are saved in folder `configs/`.

# Embeddings
download esm model from https://github.com/facebookresearch/esm.
## Unzip repository and Configure the environment
## create a new folder `esm2`
$ mkdir embeddings
## generate ESM-2 embeddings 
```
cd scripts
python extract.py esm2_t33_650M_UR50D.pt ${dataset}.fasta PepBAN/esm2/${dataset}/ --include mean
```
`${dataset}` could either be `BioGRID`, `STRING`, `camp` and `hsm`.

# Run PepBAN on Our Data to Reproduce Results

To train PepBAN, you can directly run the following command. `${dataset}` could either be `BioGRID`, `STRING`, `camp` and `hsm`.

```
python main.py  --data ${dataset} 
```


