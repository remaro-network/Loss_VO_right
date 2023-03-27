import unittest
import torch
import os
from PIL import Image
from utils.util import map_fn, operator_on_dict
import numpy as np

from model.deepvo.deepvo_model import DeepVOModel
from model.loss_functions.pose_losses import mse_euler_pose_loss, se3_chordal_loss
from model.metric_functions.vo_metrics import mse_euler_pose_metric, mse_euler_rotation_metric
from trainer.deepvo_trainer import to_device
from data_loader.data_loaders import SingleDataset
from torch.utils.data import DataLoader

class DeepVOModuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._h = 384
        cls._w = 512
        cls._device = torch.device('cuda:0')

        test_sequence="SeaFloor/track1"
        cfg_dir=os.path.join(os.getcwd(),"configs","data_loader","MIMIR", test_sequence+".yml")
        cls._dataloader = DataLoader(SingleDataset(cfg_dir),batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    def test_deepvoForward(self):
        model = DeepVOModel(batchNorm = True, checkpoint_location=["saved/checkpoints/FlowNet2_checkpoint.pth.tar"],
                conv_dropout = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5],image_size = (480,720), rnn_hidden_size=1000,
                rnn_dropout_out=.5, rnn_dropout_between=0)
        
        data_dict = next(iter(self._dataloader))
        data_dict = to_device(data_dict, self._device)
        model.to(self._device)
        out = model(data_dict)
        loss_dict = se3_chordal_loss(out)
        loss_dict = map_fn(loss_dict, torch.mean) # if loss dict, average losses
        loss = loss_dict["loss"]
        metric = mse_euler_pose_metric(out)
        metric = mse_euler_rotation_metric(out)


        loss.backward()


        

if __name__ == '__main__':
    unittest.main()