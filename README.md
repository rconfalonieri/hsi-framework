# hsi-framework

## The goal

Hyperspectral images consist of a multitude of spectral bands for each pixel. Spectral bands provide information about wavelengths, which may cover a larger spectrum of what a human eye may see. The classification task using hyperspectral images is usually addressed by taking into account only the spectral information, whereas the spatial information is ignored. To bridge this gap, this project proposes a CNN-based end-to-end framework for the classification of hyperspectral images. The proposed framework consists of a spatial and spectral classifier that are integrated to make the final prediction for the classification task. Each classifier is built by adapting a general image classifier, which is suitable for the classification of three-band images, to handle hyperspectral images. The framework is trained and validated on a real dataset—provided by a company working in the wood domain—for wood fungi detection. 


## Requirements

Matlab version: 9.10 (R2021a), with the following toolboxes installed:
- Computer Vision Toolbox, version 10.0
- Deep Learning Toolbox, version 14.2
- Image Processing Toolbox, version 11.3
- Statistics and Machine Learning Toolbox, version 12.1
- Signal Processing Toolbox, version 8.6
- Text Analytics Toolbox, version 1.7
- Parallel Computing Toolbox, version 7.4

## Script description

The repository contains 4 main scripts, each of which is responsible for a different task.
- *spatial_main.m*: implementation of the spatial branch, evaluated on the test set
- *main_full.m*: implementation of the spectral branch, evaluated on the test set
-  *branches_comb_main.m*: implementation of the final combined framework, obtained by integrating the two branches and evaluated on the test set
- *framework_tuning.m*: script for tuning the final framework by training it on the train set with different levels of trainable layers

## Running the framework
- **Spatial branch**: in MATLAB, run the "*spatial_main.m*" script. This will produce two variables  *accuracy_final* and *confmat_final* correspondent to the evaluation of the spatial branch on the test set. Use this script to produce Table 2 in the paper.
- **Spectral branch**: in MATLAB, run the "*main_full.m*" script. This will produce two variables  *accuracy_final* and *confmat_final* correspondent to the evaluation of the spectral branch on the test set. Use this script to produce Table 5 in the paper.
- **Combined framework**: in MATLAB, run the "*branches_comb_main.m*" script to combine the branches and build the complete framework. This will produce two variables *accuracy_final* and *confmat_final* correspondent to the evaluation of the framework on the test set. Use this script to produce Table 6 in the paper.
- **Framework tuning**: in MATLAB, run the "*framework_tuning.m*" script to fine tune the combined framework and train it on the training set. Use the variable *opts* to manipulate the training options. This will produce two variables  *accuracy_final* and *confmat_final* correspondent to the evaluation of the tuned framework on the test set. 
