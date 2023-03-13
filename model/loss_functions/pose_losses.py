import torch
from torch import nn
from utils.conversions import rotation_matrix_to_angle_axis
from model.metric_functions.vo_metrics import mse_metric


def mse_euler_pose_loss(data_dict):
    ''' Loss function for euler angles. Note:
    data_dict["result"] is a tensor with shape (batch x sequence_len x 6)
    data_dict["poses"] is a list with len = sequence_len, each element in list
    is a tensor with shape (batch x (4x4))
    returns: average for batch and sequence of the loss'''
    estimate = data_dict["result"] # 1,1,6 - 1,3,6 
    target = data_dict["poses"] # list of seq with (batch, 4x4) 1(1x(4,4)) - 3(1x(4,4))
    
    seq_len = estimate.size()[1] # (batch, seq, dim_pose)
    
    loss_dict = dict()
    sequence_loss = 0.
    sequence_rotation_loss = 0.
    sequence_pos_loss = 0.
    for i in range (0,seq_len): # relative T, we don take first frame in seq
        # preprocessing target values
        R_kf_i = target[i][:, :3, :3]
        R_size = R_kf_i.size()
        euler_target = rotation_matrix_to_angle_axis(torch.reshape(R_kf_i,(R_size[0],3,3)))
        t_target = target[i][:,:-1,-1]

        euler_estimate = estimate[:, i, :3]
        t_estimate = estimate[:, i, 3:]

        pos_loss = mse_metric({"result":t_estimate, "target":t_target}) # already providing mean  val
        rot_loss = mse_metric({"result":euler_estimate, "target":euler_target}) # already providing mean val
   
        loss = 100. * rot_loss + pos_loss
        loss_dict[f"loss_frame_{i}"] = loss
        sequence_loss += loss
        sequence_rotation_loss += rot_loss
        sequence_pos_loss += pos_loss

    loss_dict["loss"] = sequence_loss/seq_len
    loss_dict["rotation_loss"] = sequence_rotation_loss/seq_len
    loss_dict["traslation_loss"] = sequence_pos_loss/seq_len

    return loss_dict