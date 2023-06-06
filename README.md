# baseline repo for structuring learning-based VO projects

Add the base directory of this repo to your pythonpath
> export PYTHONPATH=$PYTHONPATH:/your-path/VO_baseline




# 1. DeepVO

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