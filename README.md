# NicheTrans: Spatial-aware Cross-omics Translation
This is the *official* Pytorch implementation of "NicheTrans: Spatial-aware Cross-omics Translation". 

## Overview
![framework](overall.png)

Spatial omics technologies have revolutionized our studies on tissue architecture and cellular interactions. While spatial multi-omics offers unprecedented insights into complex biological systems, its widespread adoption is hindered by technical challenges, specialized requirements, and limited accessibility. To address these limitations, we present NicheTrans, the first spatially-aware cross-omics translation method and a flexible Transformer-based multi-modal framework. Unlike existing single-cell translation methods, NicheTrans uniquely incorporates both cellular microenvironment information and multi-modal data. We validated the advantage of NicheTrans across diverse biological cases. Through NicheTrans, we uncovered spatial multi-omics domains that were not detectable through single-omics analysis alone. Model interpretation revealed key molecular relationships, including gene programs associated with dopamine metabolism and amyloid β-associated cell states. Additionally, using translated protein markers as spatial landmarks, we quantified the spatial organization of key glial cell subtypes in the Alzheimer's Disease brain. NicheTrans represents a powerful tool for generating comprehensive spatial multi-omics insights from more accessible single-omics measurements, making multi-omics analysis more feasible for the broader research community.

## Preparation
### Installation
```bash
pip install -r requirements.txt
(Here, we list some packages crucial for model reproduction.）
```

### Prepare Datasets
(1) Spatial Multimodal Analysis (SMA) dataset of Parkinson's Disease from ['Spatial multimodal analysis of transcriptomes and metabolomes in tissues'](https://www.nature.com/articles/s41587-023-01937-y). 

(2) STARmap PLUS dataset of Alizimizer's Disease from ['Integrative in situ mapping of single-cell transcriptional states and tissue histopathology in a mouse model of Alzheimer’s disease'](https://www.nature.com/articles/s41593-022-01251-x).

(3) Human breast cancer dataset from ['High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis'](https://www.nature.com/articles/s41467-023-43458-x). 

(4) Human lymph node dataset from ['Deciphering spatial domains from spatial multi-omics with SpatialGlue'](https://www.nature.com/articles/s41592-024-02316-4).

(5) MISAR-seq dataset from ['Simultaneous profiling of spatial gene expression and chromatin accessibility during mouse brain development'](https://www.nature.com/articles/s41592-023-01884-1).

The detailed processing pipelines for the SMA and STARmap PLUS datasets are located in the '1_Data_preparation' folder.
Here, we released the processed spatial multi-omics data at ['GoogleDrive'](https://drive.google.com/drive/folders/1YKBM-N4bP6WJyQ07EmRoZI5lQl0EKFF6?usp=drive_link). Please feel free to reach out to us if the link has expired. 

## Model training & testing
For each spatial multi-omics dataset, we have prepared the Jupyter files for model training and testing separately. __Before model training & testing, please change the direction of the datasets in the 'args' folder.__

The detailed correspondence between subfigures in the manuscript and Jupyter files is shown in:
```bash
codes_figures.txt 
```

Taking the SMA dataset as an example, the training process of NicheTrans is shown in:
```bash
Tutorial 3.1 Train NicheTrans on SMA data.ipynb
```
We provide the jupyter notebook for quantitative evaluation using pcc, spcc, and qualitative visualization of the translated results. 
```bash
Tutorial 3.2 Visualize results.ipynb
```

## Batch Sweep On Linux
If you want to train with multiple hyper-parameter combinations and save the evaluation results automatically, you can use:

```bash
python train_sweep.py --dataset sma \
  --set path_img=/your/data/patches \
  --set rna_path=/your/data/sma \
  --set msi_path=/your/data/sma \
  --set max_epoch=20 \
  --grid lr=0.0003,0.0001 \
  --grid dropout_rate=0.1,0.2 \
  --grid num_experts=1,4
```

The script writes one folder per run and stores:

```bash
config.json
history.jsonl
best_model.pth
last_model.pth
metrics.json
summary.csv
summary.jsonl
```

You can also use a JSON config file:

```json
{
  "set": {
    "max_epoch": 20,
    "path_img": "/your/data/patches",
    "rna_path": "/your/data/sma",
    "msi_path": "/your/data/sma"
  },
  "grid": {
    "lr": [0.0003, 0.0001],
    "dropout_rate": [0.1, 0.2],
    "num_experts": [1, 4]
  }
}
```

Example:

```bash
python train_sweep.py --dataset sma --grid-file sweep.json
```
## Model attribution analysis
Apart from the spatial cross-omics translation, we also provided guidelines for attribution analysis in 'Tutorial 3.3', 'Tutorial 6.5', and 'Tutorial 6.6'.

'Tutorial 3.3' and 'Tutorial 6.5' are for niche-level attribution analysis across genes. 

'Tutorial 6.6' can reveal intercellular cross-omics interaction in the spatial context. 

## Contact
If you have any questions, please don't hesitate to contact us. E-mail: [zkwang00@gmail.com](mailto:zkwang00@gmail.com); [zhiyuan@fudan.edu.cn](mailto:zhiyuan@fudan.edu.cn).
