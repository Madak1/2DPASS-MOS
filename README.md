# 2DPASS-MOS

2DPASS-MOS is a Moving Object Segmentation (MOS) network based on [2DPASS](https://github.com/yanx27/2DPASS). 
It operates on 3D LiDAR point clouds, but instead of semantic segmentation, the network decomposes the scene into static and dynamic objects using data fusion.
The network takes advantage of 2D images during training, such as dense color information and fine grained texture, to provide additional information to the ponit cloud. 
After the traning, the network performs segmentation on the clean LiDAR point clouds, whithout using the images directly.

Using just one scan achieves remarkable results, but this network also provides the possibility to use multiple (sparse) LiDAR point clouds in both training and prediction to extract additional moving information. 
To create the multi-scan version of the model, the solution provided by the [4DMOS](https://github.com/PRBonn/4DMOS) was a great help during the implementation.

We want to thank the original authors for their clear implementation and great work, which has greatly helped our project.


## How it works

In case of one scan, it works the same as 2DPASS, but instead of semantic segmentation, the network performs moving object segmentation.
However, in the case of multiple scans, several internal structural changes were made.

The network first performs a sparse operation, taking only the odd (or even) points of the point cloud.
After that the network transforms the point clouds into a common point based on the current scan, and finally performs a merge.

<p align="center">
   <img src="figures/pc-merge.png" width="90%"> 
</p>

The output of the network in this form does not match the expectations of the SemanticKITTI Banchmarks, so it requires post-processing to evaluate the results.
Predictions must be evaluated for both even and odd sparse models, and then the results of these models must be combined.
The first step is to select only the points of the current scan, and then merge the even and odd results.

<p align="center">
   <img src="figures/pred-merge.png" width="90%"> 
</p>


## Installation

- Download the original [2DPASS](https://github.com/yanx27/2DPASS) then modify it as described below.
- The dependencies will be the same as for the original 2DPASS.
- The tests were run on the [SemanticKITTI](http://www.semantic-kitti.org/index.html) dataset. 

### Data Preparation

As with 2DPASS, you need to download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and the color data from the [Kitti Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and extract them into a folder.

```
./dataset/
├── ...
└── SemanticKitti/
    ├── ...
    └── dataset/
        ├── ...
        └── sequences/
            ├── 00/ # 00-10 for traning       
            │   ├── velodyne/	
            |   |	├── 000000.bin
            |   |	├── 000001.bin
            |   |	└── ...
            │   └── labels/ 
            |   |   ├── 000000.label
            |   |   ├── 000001.label
            |   |   └── ...
            |   └── image_2/ 
            |   |   ├── 000000.png
            |   |   ├── 000001.png
            |   |   └── ...
            |   calib.txt
            |   poses.txt
            ├── 08/ # for validation
            ├── 11/ # 11-21 for testing
            └── 21/
```


### MOS adaptation
- ToDo: Describe the Config changes


## Training
- ToDo: Original, just different config


## Testing
- ToDo: Original, just different config
- ToDo: Describe num_vote
- ToDo: Describe checkpoint

## Evaluation and visualization
- ToDo: LMNET
### How to evaluate
### How to visualize


## Try 2 Frame version (Beta)
- ToDo: Describe 2F version

## Pretrain models and Results
- ToDo: Uploade models
- ToDo: Show results

## License
This repository is released under HUN-REN SZTAKI License (see LICENSE file for details).
- ToDo: Add LICENSE file
