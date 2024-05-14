# Memorization_UMAP
- Model : ResNet-18
- Dataset : CIFAR-10
- Permuted Rate : 50%
- epoch : 100
- learning rate : 1e-3

## Dataset Download
- https://drive.google.com/drive/u/0/folders/11drj9K46RaRBKTl5dVXHp13BWxiLdx_4

## Setup
- python : 3.8.19
- pytorch : 1.7.1+cu110
- cuda : 11.2

## Usage
- train.py : Training
- model.py : resnet-18 block
- dataloader.py : load data
- distribution.py : Check the distribution of randomly labeled classes
- tsne-img.py : t-SNE visualization
- umap_img.py : UMAP visualization

## [Reference]
Stephenson, Cory, et al. "On the geometry of generalization and memorization in deep neural networks." arXiv preprint arXiv:2105.14602 (2021).
