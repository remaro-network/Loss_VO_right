import unittest
import torch
import os
from PIL import Image
from utils.util import map_fn, operator_on_dict, LossWrapper
import numpy as np
from model.loss_functions import pose_losses
from model.deepvo.deepvo_model import DeepVOModel
from model.loss_functions.pose_losses import mse_euler_pose_loss, se3_chordal_loss
from model.metric_functions.vo_metrics import mse_euler_pose_metric, mse_euler_rotation_metric
from trainer.deepvo_trainer import to_device
from data_loader.data_loaders import SingleDataset, MultiDataset
from torch.utils.data import DataLoader
from base.logger import VOLogger
from utils.loadconfig import ConfigLoader


class DeepVOModuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_loader = ConfigLoader()
        config = config_loader.merge_cfg('configs/test/deepvo_se3/mimir.yml')
        cls.device = config.n_gpu
        # Model setup
        cls.model = DeepVOModel(batchNorm=config.model.args.batchNorm,checkpoint_location=config.model.args.checkpoint_location,
                            conv_dropout=config.model.args.conv_dropout, image_size = config.model.args.image_size, 
                            rnn_hidden_size=config.model.args.rnn_hidden_size,
                            rnn_dropout_out=config.model.args.rnn_dropout_out,rnn_dropout_between=config.model.args.rnn_dropout_between)
        cls.model.to(cls.device)
        # Loss config
        cls.loss_cfg = config.loss
        # Trainer config
        cls.trainer_args = config.trainer
        cls.metrics = cls.trainer_args.metrics
        # Writer
        cls.writer = VOLogger(log_dir=config.log_dir, log_step=config.log_step)
        # Optimizer
        cls.optimizer = getattr(torch.optim,config.optimizer['type'])(cls.model.parameters(),**config.optimizer.args)
        # Data loader
        cfg_dirs = [os.path.join(os.getcwd(),"configs","data_loader","MIMIR", test_sequence+".yml") for test_sequence in config.data_loader.sequences]
        cls.data_loader = DataLoader(MultiDataset(cfg_dirs),batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        # Model checkpoint saving
        cls.checkpoint_dir = config.trainer.save_dir

    def test_10epochDeepvoMemorization(self):
        for epoch in range(10):
            total_loss = 0
            total_loss_dict = {}
            total_metrics = np.zeros(len(self.metrics))	

            lossfcn = getattr(pose_losses, self.loss_cfg)

            for batch_idx, data in enumerate(self.data_loader):
                # Every data instance is a pair of input data + target result
                data = to_device(data, self.device)
                
                # Gradients must be zeroed for every batch
                self.optimizer.zero_grad()

                # Make predictions for this batch
                outputs = self.model(data)

                # Compute the loss and its gradients
                loss_dict = lossfcn(outputs)
                print(loss_dict)
                loss_dict = map_fn(loss_dict, torch.mean) # if loss dict, average losses
                loss = loss_dict["loss"]
    
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()

                loss_dict = map_fn(loss_dict, torch.detach)

                total_loss += loss.item()
                total_loss_dict = operator_on_dict(total_loss_dict, loss_dict, lambda x, y: x + y)

                # print(total_loss_dict)
                self.writer.log_dictionary(total_loss_dict,len(self.data_loader),batch_idx,epoch,'train')


                if batch_idx == 10:
                    break

            arch = type(self.model).__name__
            state = {
                'arch': arch,
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }
            filename = os.path.join(os.getcwd(),self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            print('saving checkpoint in ',filename)
            torch.save(state,filename)


        

if __name__ == '__main__':
    unittest.main()