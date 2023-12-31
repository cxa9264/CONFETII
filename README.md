# Contrast, Stylize and Adapt: Unsupervised Contrastive Learning Framework for Domain Adaptive Semantic Segmentation

## Overview

To overcome the domain gap between synthetic and real-world datasets, unsupervised domain adaptation methods have been proposed for semantic segmentation. Majority of the previous approaches have attempted to reduce the gap either at the pixel or feature level, disregarding the fact that the two components interact positively. To address this, we present **CON**trastive **FE**a**T**ure and p**I**xel alignment (CONFETI) for bridging the domain gap at both the pixel and feature levels using a unique contrastive formulation. We introduce well-estimated prototypes by including category-wise cross-domain information to link the two alignments: the pixel-level alignment is achieved using the jointly trained style transfer module with the **prototypical semantic consistency**, while the feature-level alignment is enforced to cross-domain features with the **pixel-to-prototype contrast**. Our extensive experiments demonstrate that our method outperforms existing state-of-the-art methods using DeepLabV2.

## Setup environment

We use the mmcv=1.3.7 and mmsegmentation=0.16.0. 

```
conda create -n confeti python=3.8
conda activate confeti
pip install -r requirements.txt
```

## Testing

Download checkpoints and config files from [GoogleDrive](https://drive.google.com/drive/folders/1CaClev_jycGgwrlgdVrqh_qSIODEPsEE?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1_l8x-Yd80wFrLqVD9_Vd9A) (password: 74gd).
 
```python
python -m tools.test <cfg pth> <ckpt pth>
```

## Train

### Dataset preparation

Download [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/), [Cityscapes](https://www.cityscapes-dataset.com/) datasets and [SYNTHIA](https://synthia-dataset.net/downloads/) dataset.

Extract datasets to `data` folder. The folder structure should look like this:

```
data
├── cityscapes
│   ├── gtFine
│   ├── leftImg8bit
├── gta5
│   ├── images
│   └── labels
└── synthia
    ├── GT
    └── RGB
```

### Training
```python
python run_experiments.py --config $PATH_TO_CONFIG_FILE
```
