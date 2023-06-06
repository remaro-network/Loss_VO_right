# baseline repo for structuring learning-based VO projects
Add the base directory of this repo to your pythonpath

export PYTHONPATH=$PYTHONPATH:/home/olaya/dev/VO_baseline

Flownet weights : https://github.com/dancelogue/flownet2-pytorch

# DeepVO

Experiment as in the original paper, based in sequence 00-10, as they are the ones labelled with pose ground truth.
- Training sequences: 00, 02, 08, 09
- Testing sequences: 03, 04, 05, 06, 07, 10
Optimiser: Adgrad
Epochs: 200
learning rate: 0.001
Stacking two images (seq length = 2)