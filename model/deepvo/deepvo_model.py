import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
import numpy as np

class DeepVOModel(nn.Module):
    def __init__(self, batchNorm = True, checkpoint_location=None, clip = None, image_size = (480,640),
                 conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2), output_shape = 6,
                 rnn_hidden_size = 1000, rnn_dropout_out = 0.5, rnn_dropout_between = 0):
        """
        :param pretrain_mode: Which pretrain mode to use:
            0 / False: Run full network.
            1 / True: Only run depth module. In this mode, dropout can be activated to zero out patches from the
            unmasked cost volume. Dropout was not used for the paper.
            2: Only run mask module. In this mode, the network will return the mask as the main result.
            3: Only run depth module, but use the auxiliary masks to mask the cost volume. This mode was not used in
            the paper. (Default=0)
        :param output_shape: length for pose tensor. (Default=6)
        :param pretrain_dropout: Dropout rate used in pretrain_mode=1. (Default=0)
        :param cv_patch_size: Patchsize, over which the ssim errors get averaged. (Default=3)
        :param checkpoint_location: Load given list of checkpoints. (Default=None)
        :param mask_cp_loc: Load list of checkpoints for the mask module. (Default=None)
        :param depth_cp_loc: Load list of checkpoints for the depth module. (Default=None)
        """
        super().__init__()

        # CNN
        self.batchNorm = batchNorm
        self.clip = clip
        self.rnn_hidden_size = 1000
        self.conv1   = self.conv(self.batchNorm,   6,   64, kernel_size=7, stride=2, dropout=conv_dropout[0])
        self.conv2   = self.conv(self.batchNorm,  64,  128, kernel_size=5, stride=2, dropout=conv_dropout[1])
        self.conv3   = self.conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, dropout=conv_dropout[2])
        self.conv3_1 = self.conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=conv_dropout[3])
        self.conv4   = self.conv(self.batchNorm, 256,  512, kernel_size=3, stride=2, dropout=conv_dropout[4])
        self.conv4_1 = self.conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=conv_dropout[5])
        self.conv5   = self.conv(self.batchNorm, 512,  512, kernel_size=3, stride=2, dropout=conv_dropout[6])
        self.conv5_1 = self.conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=conv_dropout[7])
        self.conv6   = self.conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=conv_dropout[8])

        # Comput the shape based on diff image size
        __tmp = torch.zeros(1, 6, image_size[1], image_size[0])
        __tmp = self.encode_image(__tmp)
        # RNN
        self.rnn = nn.LSTM(
                    input_size = int(np.prod(__tmp.size())),
                    hidden_size = rnn_hidden_size, 
                    num_layers = 2, 
                    dropout = rnn_dropout_between, 
                    batch_first = True)
        self.rnn_drop_out = nn.Dropout(rnn_dropout_out)
        self.linear = nn.Linear(in_features=rnn_hidden_size, out_features=output_shape)
        # initalization from https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/model.py
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                kaiming_normal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                n = m.bias_hh_l0.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l0.data[start:end].fill_(1.0)

                kaiming_normal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l1.data[start:end].fill_(1.0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        # Checkpoint (if any)
        self.checkpoint_location = checkpoint_location
        if self.checkpoint_location is not None:
            pretrained_model = torch.load(checkpoint_location[0], map_location="cpu")
            current_state_dict = self.state_dict()
            update_state_dict = {}
            if pretrained_model['arch']=='DeepVOModel':
                for k, v in pretrained_model["model_state_dict"].items():
                    if k in current_state_dict.keys():
                        update_state_dict[k] = v
            else:
                for k, v in pretrained_model["state_dict"].items():
                    if k in current_state_dict.keys():
                        update_state_dict[k] = v
            current_state_dict.update(update_state_dict)
            self.load_state_dict(current_state_dict)

    def forward(self, data_dict):
        keyframe = data_dict["keyframe"]
        frames = data_dict["frames"]
        batch_size = keyframe.size(0)
        seq_len = len(frames)# + 1
        # print('keyframe shape',keyframe.shape)
        # print('frames shape', len(frames))
        # print('single frame shape',frames[0].shape)

        stacked_imgs = []
        stacked_imgs.append(torch.cat((keyframe,frames[0]),dim = 1))
        for i in range(len(frames)-1):
            stacked_imgs.append(torch.cat((frames[i],frames[i+1]),dim = 1))
        
        frames_stack = torch.stack(stacked_imgs, dim=1) # stack seqs, not batches!
        # print("seq_stack Shape",frames_stack.size())

        frames_stack = frames_stack.view(batch_size * seq_len, frames_stack.size(2), frames_stack.size(3), frames_stack.size(4))
        # print("1st view seq_stack Shape",frames_stack.size())

        flow_stack = self.flow(frames_stack)
        # print("after flow Shape",frames_stack.size())
        flow_stack = flow_stack.view(batch_size, seq_len, -1)
        # print("after reshape Shape",frames_stack.size())
        if not self.training:
            if not "hidden" in data_dict:
                encoded_stack, h = self.rnn(flow_stack)
            else:
                encoded_stack, h = self.rnn(flow_stack, data_dict["hidden"])
            data_dict["hidden"] = h
        else:
            encoded_stack, _ = self.rnn(flow_stack)
            
        encoded_stack = self.rnn_drop_out(encoded_stack)
        pose_stack = self.linear(encoded_stack)
        data_dict["result"] = pose_stack
        data_dict["target"] = data_dict["poses"] # for compatibility with losses notation
        return data_dict

    def flow(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.conv4(x)
        x = self.conv4_1(x)
        x = self.conv5(x)
        x = self.conv5_1(x)
        x = self.conv6(x)
        return x

    def conv(self, batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
        if batchNorm:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)#, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)#, inplace=True)
            )
    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6


