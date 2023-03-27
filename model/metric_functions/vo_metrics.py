import torch
from torch import nn
from utils.conversions import rotation_matrix_to_angle_axis, se3_exp_map, so3_exp_map

def SE3_chordal_metric_old(data_dict, t_weight = 1, orientation_weight = 1):
    so3_chordal = SO3_chordal_metric(data_dict)
    tnorm = vector_distance(data_dict)
    SE3_c = orientation_weight*so3_chordal + t_weight*tnorm
    return SE3_c

def SE3_chordal_metric(T_estimate,T_target, t_weight = 1, orientation_weight = 1):
    R_estimate = T_estimate[:, :3, :3]
    R_target = T_target[:, :3, :3]
    t_estimate = T_estimate[:,:-1,-1]
    t_target = T_target[:,:-1,-1]
    so3_chordal = SO3_chordal_metric(R_estimate,R_target)
    tnorm = vector_distance(t_estimate,t_target)
    SE3_c = orientation_weight*so3_chordal + t_weight*tnorm
    return SE3_c

def SO3_chordal_metric(R_estimate,R_target):
    """
    Compute the geodesic distance between two rotation matrices.
    """

    R_diff = torch.matmul(R_estimate,R_target.mT) - torch.eye(3)
    R_diff_norm = torch.linalg.matrix_norm(R_diff, ord='fro', dim=(- 2, - 1))**2
    R_diff_mean = torch.mean(R_diff_norm,-1,True) # mean for all batches
    metric = torch.mean(R_diff_norm,-1,True) # mean for all batches

    return metric

def vector_distance(v_estimate,v_target):

    v_diff = v_target - v_estimate
    v_diff_norm = torch.norm(v_diff, p=2, dim=1)

    return v_diff_norm

def SO3_chordal_metric_old(data_dict):
    """
    Compute the geodesic distance between two rotation matrices.
    """
    estimate = data_dict["result"] # 1,1,6 - 1,3,6 
    target = data_dict["poses"] # list of seq with (batch, 4x4) 1(1x(4,4)) - 3(1x(4,4))
    seq_len = estimate.size()[1] # (batch, seq, dim_pose
    
    sequence_metric = 0.

    for i in range (0,seq_len): # relative T, we don take first frame in seq
        # convert se3 to SE3
        R_kf_i_target = target[i][:, :3, :3]
        R_kf_i_estimate = so3_exp_map(estimate[:, i, :3])
        R_diff = torch.matmul(R_kf_i_target,R_kf_i_estimate.mT) - torch.eye(3)
        R_diff_norm = torch.linalg.matrix_norm(R_diff, ord='fro', dim=(- 2, - 1))**2
        R_diff_mean = torch.mean(R_diff_norm,-1,True) # mean for all batches
        metric = R_diff_mean
        sequence_metric += metric

    sequence_metric/=seq_len
    return sequence_metric

def vector_distance_old(data_dict):
    estimate = data_dict["result"] # 1,1,6 - 1,3,6 
    target = data_dict["poses"] # list of seq with (batch, 4x4) 1(1x(4,4)) - 3(1x(4,4))
    seq_len = estimate.size()[1] # (batch, seq, dim_pose
    
    sequence_metric = 0.

    for i in range (0,seq_len): # relative T, we don take first frame in seq
        # convert se3 to SE3
        t_kf_i_target = target[i][:,:-1,-1]
        t_kf_i_estimate = estimate[:, i, 3:]

        t_diff = t_kf_i_target - t_kf_i_estimate
        t_diff_norm = torch.norm(t_diff, p=2, dim=1)

        metric = t_diff_norm
        sequence_metric += metric

    sequence_metric/=seq_len
    return sequence_metric

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