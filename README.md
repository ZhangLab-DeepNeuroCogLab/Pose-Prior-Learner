# Pose Prior Leaner

This repository contains the official implementation of the ICLR 2026 paper: 

[Pose Prior Learner: Unsupervised Categorical Prior Learning for Pose Estimation](https://arxiv.org/pdf/2410.03858.pdf)

Ziyu Wang, Shuangpeng Han, Mengmi Zhang

## Environment Setup
The basic environment contains these packages:
- Python 3.12.9
- torch 2.7.0
- torchvision 0.22.0
- pytorch-lightning 2.5.0

Other dependencies can be installed as needed.

## Dataset
The [Taichi](https://github.com/AliaksandrSiarohin/motion-cosegmentation), [Human3.6m](http://vision.imar.ro/human3.6m/description.php), [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [11k Hands](https://sites.google.com/view/11khands), [AFHQ](https://github.com/clovaai/stargan-v2), [Horse2Zebra](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset) and [Flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) can be found on their websites.
We provide the Youtube Dog Video dataset [here](https://drive.google.com/drive/u/0/folders/1J3NWlrrVtgHMgHfFBMhEniqHnyHbl2Zo).

## Pre-trained Models
he pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/u/0/folders/1nh9HSDwN3BZP3XDQDiXXHy7X14DsqPuA).

## Training & Testing
To train the model from scratch, please follow the steps below:
- Modify the ``DATA_DIR`` in ``dataset/xxx.py`` to your own.
- Run the command as shown in the following example. The input parameter is the ID of your GPUs.
```
sh script/train_hands.py
```

To test the model:
- Modify the ``DATA_DIR`` in ``dataset/xxx.py`` to your own.
- Run the command as shown in the following example. The input parameter is the ID of your GPUs.
```
sh script/test_hands_voc.py
```
## Citation
If you find our paper and/or code helpful, please cite:
```
@article{wang2024pose,
  title={Pose Prior Learner: Unsupervised Categorical Prior Learning for Pose Estimation},
  author={Wang, Ziyu and Han, Shuangpeng and Zhang, Mengmi},
  journal={arXiv preprint arXiv:2410.03858},
  year={2024}
}
```