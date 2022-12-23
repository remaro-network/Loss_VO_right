import unittest
import torch
from PIL import Image

from model.releVO.releVO_model import releVO_model

class releVOModuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._h = 384
        cls._w = 512
        cls._device = torch.device('cuda:0')
    
    def test_relevoForward(self):
        model = releVO_model(batchNorm = True, checkpoint_location=["saved/models/flownet/FlowNet2_checkpoint.pth.tar"],
                conv_dropout = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5],image_size = (52,80), rnn_hidden_size=1000,
                rnn_dropout_out=.5, rnn_dropout_between=0)
        
        # data_dict = self._dataset.__getitem__(4)
        # data_dict = next(iter(self._dataloader))
        # data_dict = to(data_dict, self._device)
        model.to(self._device)
        out = model(data_dict)
        loss_dict = mse_euler_pose_loss(out)
        loss_dict = map_fn(loss_dict, torch.mean) # if loss dict, average losses
        loss = loss_dict["loss"]
        metric = mse_euler_pose_metric(out)
        metric = mse_euler_rotation_metric(out)


        loss.backward()

if __name__ == '__main__':
    unittest.main()