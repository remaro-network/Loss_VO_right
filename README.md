# baseline repo for structuring learning-based VO projects

Add the base directory of this repo to your pythonpath
> export PYTHONPATH=$PYTHONPATH:/your-path/VO_baseline


# 1. Quickstart

## 1.1 Generate configs for data loaders

VO_baseline is designed for an easy deployment in your computer.
It has data loaders for KITTI, TUM-RGBD, EuRoC, Aqualoc and MIMIR. This repository expects your to have those datasets under the same folder, and under the names KITTI, TUM, EuRoC, Aqualoc and MIMIR.

If you have it like that, you can go to the file under `/scripts/generate_configs.sh`
This script will automatically generate configs to load the aforementioned datasets. Within that file, you will need to modify the following variables to fit your setup:
```
# Set conda environment
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE"/etc/profile.d/conda.sh
conda activate olayaenv
```
After the `conda activate` directive, put the name of your code environment. Mine is `olayaenv`.
```
# Set directories
datasetRoot=$HOME/Datasets
Datasets="KITTI MIMIR Aqualoc/Archaeological_site_sequences EuRoC TUM"

```
In `datasetRoot` you need to put the path to your folder containing all datasets. The variable `Datasets` indicates for which datasets available in the pipeline we want to create configs.

# 2. Networks
## 2.1. DeepVO

> Wang, S., Clark, R., Wen, H., & Trigoni, N. (2017, May). Deepvo: Towards end-to-end visual odometry with deep recurrent convolutional neural networks. In 2017 *IEEE international conference on robotics and automation (ICRA)* (pp. 2043-2050). IEEE.

![](https://github.com/olayasturias/VO_baseline/blob/deepvo/media/deepvo.png?raw=true)

It is based on a pretrained Flownet. You can find the Flownet weights at : https://github.com/dancelogue/flownet2-pytorch

All the DeepVO models have been trained under the same experimental setup as follows:

| Train | Validation | Test |
|-------------|-------|-------|
|00, 02-06,08 | 07,09 | 01,10 |


- Optimiser: Adgrad
    - Epochs: 200
    - learning rate: 0.001
    - Stacking two images (seq length = 2)