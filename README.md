# Baseline repo for structuring learning-based VO projects

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQIymvK2fEydLiqzGboQLi7OrgsttKx6E_TpQUxyPz5DqUXVdM4ep27F-YjEl46axQlQC4&usqp=CAU)


# 1. Quickstart :hedgehog:

Add the base directory of this repo to your pythonpath
> export PYTHONPATH=$PYTHONPATH:/your-path/VO_baseline

## 1.1 Generate configs for data loaders (Automatically!)

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


### 1.1.1 Edit the data loader
Although the configs are meant to work out of the box, you might want to edit some of the parameters for your custom setup. You can either manually edit each of the generated configs, or edit the config called `default_configuration.yml`. 

`default_configuration.yml` is a template available for each dataset from which the configs for each track are generated.
If you edit that file, when you generate the configs running `/scripts/generate_configs.sh`, those changes will be applied to all the generated configs. The parameters you might want to edit are:
- `crop_box`: wether to crop a subset of the images or not. If empty, the data loader retrieves the full image. If not empty:
    - [ow, oh, bw, bh] origin, width, and height of the box, respectively.
    - [random, random, bw, bh] same but the origin coordinates of the bounding box are generated randomly.
    - Example: given an image with width w and height h, crop the box [ow, oh, bw, bh]

        ```
                0                    w
            0  +---------------------+->
                |    ow          bw  |
                | oh +-----------+   |
                |    |           |   |
                |    |   image   |   |
                |    |           |   |
                | bh +-----------+   |
                |                    |
            h   +--------------------+
                |
                v
        ```

## 1.2 Generate configs for your experiment under the existing models

TBD

## 1.3 Setup your own VO model

TBD

## 1.4 Setup your loss functions

TBD

## 1.5 Unit tests are your friends

TBD


# 2. Networks :eye:
## 2.1. DeepVO

> Wang, S., Clark, R., Wen, H., & Trigoni, N. (2017, May). Deepvo: Towards end-to-end visual odometry with deep recurrent convolutional neural networks. In 2017 *IEEE international conference on robotics and automation (ICRA)* (pp. 2043-2050). IEEE.

![](https://github.com/olayasturias/VO_baseline/blob/deepvo/media/deepvo.png?raw=true)

It is based on a pretrained Flownet. You can find the Flownet weights at : https://github.com/dancelogue/flownet2-pytorch

Under `/configs/train` you can find several experiment configurations for DeepVO, as depicted in the figure above: `deepvo`, `deepvo_quat` and `deepvo_se3`.

### 2.1.1 deepvo
Original setup for DeepVO. Here, the output head is a 6-dimensional vector corresponding to the translation vector and the three Euler angles. Under `/configs/train/deepvo` you can find the configs for the experiments under this setup:
- `original_paper.yml`: with data explit and model configuration as proposed in the original DeepVO paper.
- `icra23.yml`: with data split as proposed by me in my ICRA workshop paper [link TBD].
- `RWzhou.yml`: with data split as proposed in [TBD].
### 2.1.2 deepvo_quat
Here, the output head is a 7-dimensional vector corresponding to the translation vector and the quaternion. Under `/configs/train/deepvo_quat` you can find the configs for the experiments under this setup:
- `original_paper.yml`: with data explit and model configuration as proposed in the original DeepVO paper.
- `icra23.yml`: with data split as proposed by me in my ICRA workshop paper [link TBD]. The loss function corresponds to the Euclidean loss
- `icra23_geodesic.yml`: with data split as proposed in [TBD]. The loss function corresponds to the geodesic loss.
### 2.1.3. deepvo_se3
Original setup for DeepVO. Here, the output head is a 6-dimensional vector corresponding to the translation vector and the three Euler angles. Under `/configs/train/deepvo` you can find the configs for the experiments under this setup:
- `original_paper.yml`: with data explit and model configuration as proposed in the original DeepVO paper.
- `icra23.yml`: with data split as proposed by me in my ICRA workshop paper [link TBD].
- `RWzhou.yml`: with data split as proposed in [TBD].


All the DeepVO models have been trained under the same experimental setup as follows:

| Train | Validation | Test |
|-------------|-------|-------|
|00, 02-06,08 | 07,09 | 01,10 |


- Optimiser: Adgrad
    - Epochs: 200
    - learning rate: 0.001
    - Stacking two images (seq length = 2)