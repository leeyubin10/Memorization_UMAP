# Memorization_UMAP
- Model : ResNet-18
- Dataset : CIFAR-10
- Permuted Rate : 50%
- epoch : 100
- learning rate : 1e-3

## Dataset Download
- https://drive.google.com/drive/u/0/folders/11drj9K46RaRBKTl5dVXHp13BWxiLdx_4

##
```Memorization_UMAP
├── cifar-10
│   ├── train
│   ├── test
│   └── val
│ 
├── train_shuffle.py
├── train.py
├── model.py
├── tsne_model.py
├── distribution.py
├── custom_checkpoint.py
├── tsne_img.py
└── umap_img.py
``` 

## Setup
- python : 3.8.19
- pytorch : 1.7.1+cu110
- cuda : 11.2

## Usage
- train_shuffle.py : Generate random labels according to a specified ratio
- train.py : training a neural network model on CIFAR-10 dataset
- model.py : resnet-18 block
- tsne_model.py : Replace the fully connected (fc) layer with the Identity layer
- distribution.py : Check the distribution of randomly labeled classes
- custom_checkpoint.py : This script removes unexpected keys from checkpoint files
- tsne-img.py : t-SNE visualization
- umap_img.py : UMAP visualization

## Implementation
1. Reconstruct the training dataset
```
python train_shuffle.py
```  
2. train
```
python train.py
```
3. custom checkpoint
```
python custom_checkpoint.py
```
4. test & visualization
```
python tsne_img.py
python umap_img.py
```

## [Reference]
Stephenson, Cory, et al. "On the geometry of generalization and memorization in deep neural networks." arXiv preprint arXiv:2105.14602 (2021).
