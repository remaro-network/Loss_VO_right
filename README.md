

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


# 3. Inference

So you have your model and now want to see which results it's inferring?
For that, you can execute the script under `visualization/<algorithm_name>_inference_results.py`. The inference scripts that are available so far are:
- DeepVO 

    `visualization/deepvo_inference_results.py`. For this script, you need to set the variables:
    - test_sequences: a list of all the sequences that you want to infer on.
    - experiment: name of your experiment. This will be used to save the results.
    - models: list of models you want to infer with.
    The results (that is, the obtained trajectories) will be saved under `visualization/<experiment-name>` with the naming convention `<sequence_name>_<model-name>.csv`. Additionally, the trajectories are also saved in the tum format compatible with the evo tools as `evo_<sequence_name>_<model-name>.csv`.