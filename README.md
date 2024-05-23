# Memorization_t-SNE
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
│   ├── train   ## permuted or original
│   ├── test
│   └── val
│
├── original_label
│   ├── airplane
│   ├── ...
│   └── truck
│
└── random_label
    ├── airplane
    ├── ...
    └── truck
``` 

## Setup
- python : 3.8.19
- pytorch : 1.7.1+cu110
- cuda : 11.2

## Usage
- data_shuffle.py : Generate random labels according to a specified ratio
- train.py : training a neural network model on CIFAR-10 dataset
- model.py : resnet-18 block
- tsne_model.py : Replace the fully connected (fc) layer with the Identity layer
- distribution.py : Check the distribution of randomly labeled classes
- custom_checkpoint.py : This script removes unexpected keys from checkpoint files
- tsne-img.py : t-SNE visualization
- tsne-img_permuted.py : t-SNE visualization by distinguishing between random_label and original_label
- umap_img.py : UMAP visualization

## Implementation
1. Reconstruct the training dataset
```
python data_shuffle.py
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

## Results
<img src="https://github.com/leeyubin10/Memorization_t-SNE/assets/68275474/b21d3663-5414-4bb5-9b2e-ba91316e2caa" width="400" height="400"/>

## [Reference]
Stephenson, Cory, et al. "On the geometry of generalization and memorization in deep neural networks." arXiv preprint arXiv:2105.14602 (2021).
