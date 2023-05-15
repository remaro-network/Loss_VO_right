import torch
from torch import nn
from utils.conversions import rotation_matrix_to_angle_axis, se3_exp_map, so3_exp_map

def qinv(q: torch.Tensor):
    """
    Invert quaternions
    Expect a tensor of shape (*, 4)
    Returns a tensor of shape (*, 4)
    """
    ori_shape = q.shape
    q = q.contiguous().view(-1, 4)
    q_norm = torch.norm(q, p=2, dim=1)
    q_inv = q.clone().contiguous().view(-1, 4)
    q_inv[:, 1:] *= -1
    q_inv /= q_norm.unsqueeze(1)
    return q_inv.reshape(*ori_shape)


def quaternion_geodesic_distance(q1,q2):
    """
    calculate geodesic distance from q1 to q2 in the form of delta cosine value
    :param q1: (*, 4)
    :param q2: (*, 4)
    :return: (*)
    """
    assert q1.shape == q2.shape
    assert q1.shape[-1] == q2.shape[-1] == 4 or q1.shape[-1] == q2.shape[-1] == 3

    ori_shape = q1.shape
    q1 = q1.contiguous().view(-1, q1.shape[-1])
    q2 = q2.contiguous().view(-1, q2.shape[-1])

    if ori_shape[-1] == 4:
        q1_inv = qinv(q1)
        terms = torch.bmm(q1_inv.view(-1, 4, 1), q2.view(-1, 1, 4))
        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        w = torch.clamp(w, -1, 1)
        distance = torch.sub(1, w).view(ori_shape[:-1])
    else:
        q1_inv = q1.clone()
        q1_inv[:, 1:] *= -1
        terms = q1_inv * q2
        w = terms[:, 0] - terms[:, 1] - terms[:, 2]
        w = torch.clamp(w, -1, 1)
        distance = torch.sub(1, w).view(ori_shape[:-1])
    return distance

def quaternion_distance_metric(q1, q2):
    """ compute the quaternion distance between two vectors """
    plus_dist = torch.linalg.vector_norm(q1+q2, dim = 1) # norm for each vector in batch
    minus_dist = torch.linalg.vector_norm(q1-q2, dim = 1) # idem 
    min_dist = minus_dist.clone()
    for i in range(min_dist.shape[0]):
        if min_dist[i] > plus_dist[i]:
            min_dist[i] = plus_dist[i]
    return torch.mean(min_dist,-1,True)

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
    device = R_estimate.get_device()
    R_diff = torch.matmul(R_estimate,R_target.mT) - torch.eye(3, device=device)
    R_diff_norm = torch.linalg.matrix_norm(R_diff, ord='fro', dim=(- 2, - 1))**2
    R_diff_mean = torch.mean(R_diff_norm,-1,True) # mean for all batches
    metric = torch.mean(R_diff_norm,-1,True) # mean for all batches

    return metric

def vector_distance(v_estimate,v_target):

    v_diff = v_target - v_estimate
    v_diff_norm = torch.norm(v_diff, p=2, dim=1)

    return v_diff_norm

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