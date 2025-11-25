# Cluster Backup Project

This repository contains tools and modules designed to implement vulnerability detection (VD) through mixture of experts (MoE).
The project is organized into four main components, each serving a specific purpose.

## Project Structure
```
cluster_backup/
│  
├── cluster/      - Clustering algorithms and data grouping utilities  
├── JGNNdet/      - Graph Neural Network (GNN) tool for VD  
├── LineVul/      - Line-based vulnerability detection tool for VD
├── router/        - Vulnerability Routing models  
├── .gitignore  
└── README.md  
```
## Instructions

Each folder contains its own instructions on how to install and use the module.  
Please navigate to the folder you are interested in and follow the provided guidelines.
It is recommended to follow the following order: cluster -> LineVul -> router -> JGNNdet


## Datasets
This brief description shows how the datasets within the project are divided into various folders.
To download and view the complete datasets:

```
gdown https://drive.google.com/uc?id=154ZaiotSgg7FHqHauywCn_sSa8Iy3pY8
unzip datasets.zip
cd datasets
```

Below is a description of the main datasets contained in the folder and how they are used in the experiment:
```
datasets/
│
├── 🟨 c/
│   ├── 📁 experts_datasets/
│   │   ├── 📁 experts_clusters
│   │   │   ├── c_train_cluster_0.csv
│   │   │   ├── c_test_cluster_0.csv               ← C datasets to train single experts based on clusters
│   │   │   ├── c_val_cluster_0.csv
│   │   │   └── ...
│   │   ├── 📁 experts_datasets_reduced
│   │   │   ├── processed_train_CWE-284-reduced.csv
│   │   │   ├── processed_test_CWE-284-reduced.csv ← C datasets to train balanced experts based on CWE 
│   │   │   ├── processed_val_CWE-284-reduced.csv
│   │   │   └── ...
│   │   ├── c_processed_train_CWE-284.csv
│   │   ├── c_processed_test_CWE-284.csv           ← C datasets to train single experts based on CWE
│   │   ├── c_processed_val_CWE-284.csv
│   │   └── ...
│   ├── 📁 router_datasets/
│   │   ├── 📁 router_clusters/
│   │   │   ├── new_c_processed_train.csv
│   │   │   ├── new_c_processed_test.csv           ← C datasets to train router based on clusters
│   │   │   └── new_c_processed_val.csv
│   │   ├── processed_train_reduced_mean.csv
│   │   ├── processed_test_reduced_mean.csv        ← C datasets to train balanced router based on CWE 
│   │   └── processed_val_reduced_mean.csv
│   ├── c_processed_cleaned.csv
│   ├── c_processed_train.csv
│   ├── c_processed_test.csv                       ← C datasets to train monolithic model
│   └── c_processed_val.csv
│
├── 🟦 java/
│   ├── 📁 experts_datasets/
│   │   ├── 📁 experts_clusters
│   │   │   ├── j_train_cluster_0.csv
│   │   │   ├── j_test_cluster_0.csv               ← Java datasets to train single experts based on clusters
│   │   │   ├── j_val_cluster_0.csv
│   │   │   └── ...
│   │   ├── 📁 random_experts
│   │   │   ├── j_equal_train_1.csv
│   │   │   ├── j_equal_test_1.csv                 ← Java datasets to train random experts 
│   │   │   ├── j_equal_val_1.csv
│   │   │   └── ...
│   │   ├── j_processed_train_CWE-284.csv
│   │   ├── j_processed_test_CWE-284.csv           ← Java datasets to train single experts based on CWE 
│   │   ├── j_processed_val_CWE-284.csv
│   │   └── ...
│   ├── 📁 router_datasets/
│   │   └── 📁 router_clusters/
│   │       ├── new_j_processed_train.csv
│   │       ├── new_j_processed_test.csv           ← Java datasets to train router based on clusters
│   │       └── new_j_processed_val.csv
│   ├── j_processed_cleaned.csv
│   ├── j_processed_train.csv
│   ├── j_processed_test.csv                       ← Java datasets to train monolithic model
│   └── j_processed_val.csv
│
├── 🟧 python/
│   ├── 📁 p_experts_datasets/              
│   │   ├── 📁 experts_clusters             
│   │   │   ├── p_train_cluster_0.csv
│   │   │   ├── p_test_cluster_0.csv               ← Python datasets to train single experts based on clusters
│   │   │   ├── p_val_cluster_0.csv
│   │   │   └── ...
│   │   ├── p_processed_train_CWE-284.csv
│   │   ├── p_processed_test_CWE-284.csv           ← Python datasets to train single experts based on CWE 
│   │   ├── p_processed_val_CWE-284.csv     
│   │   └── ...
│   ├── 📁 p_router_datasets/
│   │   └── 📁 router_clusters/
│   │       ├── new_p_processed_train.csv
│   │       ├── new_p_processed_test.csv           ← Python datasets to train router based on clusters
│   │       └── new_p_processed_val.csv
│   ├── p_processed_cleaned.csv
│   ├── p_processed_train.csv
│   ├── p_processed_test.csv                       ← Python datasets to train monolithic model
│   └── p_processed_val.csv
│
└── 🟪 graph_datasets/
    ├── 📁 merged_c/
    │   ├── c_processed_CWE-284_dataset.csv        ← C datasets to single graph experts 
    │   └── ...
    ├── 📁 merged_java/
    │   ├── j_processed_CWE-284_dataset.csv        ← C datasets to single graph experts 
    │   └── ...
    └── 📁 merged_python/
        ├── p_processed_CWE-284_dataset.csv        ← C datasets to single graph experts 
        └── ...

```

