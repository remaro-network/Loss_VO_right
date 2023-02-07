import torch
from torch import nn
from utils.conversions import rotation_matrix_to_angle_axis

def mse_metric(data_dict):
    loss = nn.MSELoss(reduction='mean')
    mse_loss = loss(data_dict["result"], data_dict["target"])  
    return mse_loss

def rmse_metric(data_dict, eps=1e-6):
    rmse_loss = torch.sqrt(mse_metric(data_dict)+eps)
    return rmse_loss

def mse_euler_pose_metric(data_dict):
    estimate = data_dict["result"] 
    target = data_dict["poses"] # list of seq with (batch, 4x4)
    seq_len = estimate.size()[1] # (batch, seq, dim_pose)
    sequence_pos_loss = 0.
    for i in range (1,seq_len): # relative T, we don take first frame in seq
        # preprocessing target values
        t_target = target[i][:,:-1,-1]
        t_estimate = estimate[:, i, 3:]
        pos_loss = mse_metric({"result":t_estimate, "target":t_target})
        sequence_pos_loss += pos_loss

    return sequence_pos_loss

def mse_euler_rotation_metric(data_dict):
    estimate = data_dict["result"] 
    target = data_dict["poses"] # list of seq with (batch, 4x4)    
    seq_len = estimate.size()[1] # (batch, seq, dim_pose)    
    loss_dict = dict()
    sequence_rotation_loss = 0.
    for i in range (1,seq_len): # relative T, we don take first frame in seq
        # preprocessing target values
        R_kf_i = target[i][:, :3, :3]
        R_size = R_kf_i.size()
        euler_target = rotation_matrix_to_angle_axis(torch.reshape(R_kf_i,(R_size[0],3,3)))
        euler_estimate = estimate[:, i, :3]
        rot_loss = mse_metric({"result":euler_estimate, "target":euler_target}) 
        sequence_rotation_loss += rot_loss
    loss_dict["rotation_loss"] = sequence_rotation_loss


    return loss_dict