# A Deep Residual Network for Geometric Decontouring

<img src="./assets/paper.png" width="900px"/>

## Introduction

<img src="./assets/teaser.png"/>

Figure 1: Geometric Decontouring for grayscale image. (a) An input grayscale. (b) Convert (a) to a 3D mesh. Note that the false contours arise. (c) Our network eliminates the false contours and preserves the fine details well. (d-e) Close-ups of (b) and (c).

An example,

<img src="./assets/example.png"/>

For details, please refer to our paper [A Deep Residual Network for Geometric Decontouring](./assets/GDCNet_PG2020.pdf).

## Architecture

<img src="./assets/network.png"/>

Figure 2: Architecture of our network for geometric detail restoration from an 8-bit grayscale image. Our network consists of two convolutional layers and several residual blocks. The gray dots indicate repetition of the residual blocks. The output of our network is the predicted residual errors.

## Dataset

<img src="./assets/dataset.png"/>

Figure 3: Some samples of our Height-Grayscale Dataset. The images shown in the top row are height maps, and the images in the bottom row are grayscales. A height map encodes the geometric details, so here we render it as a surface.

- The structure tree of the dataset:
  
  ```bash
  ./Height-Grayscale-Dataset
  ├── .
  ├── gray
  ├── └── 0.png
  ├──  ...
  ├──
  ├── height
  ├── └── 0.mat
  ├──  ...
  ├──
  ├── mask
  ├── └── 0.png
  ├──  ...  
  ```


#### The dataset can be downloaded [here](https://drive.google.com/file/d/1iJMFjMDVSZ9sHRI6yvFQwFcDvYKllno1/view?usp=sharing).


## Citation

If you find our work or Dataset useful to your research, please consider citing:


```bash
@article {pg2020GDCNet,
journal = {Computer Graphics Forum},
title = {A Deep Residual Network for Geometric Decontouring},
author = {Ji, Zhongping and Zhou, Chengqin and Zhang, Qiankan and Zhang, Yu-Wei and Wang, Wenping},
volume = {39},
number = {7},
pages = {27--41},
year = {2020},
publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
ISSN = {1467-8659},
DOI = {10.1111/cgf.14124}
}
```
