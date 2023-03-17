import unittest
import torch
import os
from PIL import Image
from utils.util import map_fn, operator_on_dict, LossWrapper
import numpy as np
from model.loss_functions import pose_losses
from model.deepvo.deepvo_model import DeepVOModel
from model.loss_functions.pose_losses import mse_euler_pose_loss
from model.metric_functions.vo_metrics import mse_euler_pose_metric, mse_euler_rotation_metric
from trainer.deepvo_trainer import to_device, DeepVOTrainer
from data_loader.data_loaders import SingleDataset, MultiDataset
from torch.utils.data import DataLoader
from base.logger import VOLogger
from utils.loadconfig import ConfigLoader


class DeepVOModuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_loader = ConfigLoader()
    def test_1epochDeepvoTrainer_miscellaneous(self):
        config = self.config_loader.merge_cfg('configs/test/deepvo/miscellaneous.yml')
        trainer = DeepVOTrainer(config)
        trainer.train()
    @unittest.skip("Skipping kitti test")
    def test_1epochDeepvoTrainer_KITTI(self):
        config = self.config_loader.merge_cfg('configs/test/deepvo/kitti.yml')
        trainer = DeepVOTrainer(config)
        trainer.train()
    @unittest.skip("Skipping mimir test")
    def test_1epochDeepvoTrainer_MIMIR(self):
        config = self.config_loader.merge_cfg('configs/test/deepvo/mimir.yml')
        trainer = DeepVOTrainer(config)
        trainer.train()
    @unittest.skip("Skipping tum test")
    def test_1epochDeepvoTrainer_TUM(self):
        config = self.config_loader.merge_cfg('configs/test/deepvo/tum.yml')
        trainer = DeepVOTrainer(config)
        trainer.train()
    @unittest.skip("Skipping euroc test")
    def test_1epochDeepvoTrainer_euroc(self):
        config = self.config_loader.merge_cfg('configs/test/deepvo/euroc.yml')
        trainer = DeepVOTrainer(config)
        trainer.train()

        

if __name__ == '__main__':
    unittest.main()