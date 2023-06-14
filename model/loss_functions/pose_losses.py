import torch
from torch import nn
from utils.conversions import rotation_matrix_to_angle_axis , se3_exp_map, so3_exp_map, rotation_matrix_to_quaternion,quaternion_to_rotation_matrix, euler_angles_to_matrix
from model.metric_functions.vo_metrics import mse_metric,SE3_chordal_metric, SO3_chordal_metric,vector_distance, quaternion_distance_metric, quaternion_geodesic_distance, SO3_geodesic_distance

def quaternion_geodesic_loss(data_dict,t_weight = 1, orientation_weight = 100.):
    ''' Loss function for pose expressed as translation vector and quaternion. Note:
    data_dict["result"] is a tensor with shape (batch x sequence_len x 7)
    data_dict["poses"] is a list with len = sequence_len, each element in list
    is a tensor with shape (batch x (4x4))
    returns: average for batch and sequence of the loss'''
    estimate = data_dict["result"] # 1,1,7 - 1,3,7
    target = data_dict["poses"] # list of seq with (batch, 4x4) 1(1x(4,4)) - 3(1x(4,4))

    seq_len = estimate.size()[1] # (batch, seq, dim_pose)

    loss_dict = dict()
    sequence_pose_loss = 0.
    sequence_quaternion_loss = 0.
    sequence_t_loss = 0.
    benchmark_rotation_loss = 0.
    for i in range (0,seq_len): # relative T, we don take first frame in seq
        # preprocessing target values
        # preprocessing target values
        R_target = target[i][:, :3, :3]
        q_target = rotation_matrix_to_quaternion(R_target)
        t_target = target[i][:,:-1,-1]

        q_estimate = estimate[:, i, 3:]
        t_estimate = estimate[:, i, :3]

        quaternion_loss = quaternion_geodesic_distance(q_estimate,q_target)
        t_loss = vector_distance(t_target,t_estimate)
        pose_loss = t_weight*t_loss + orientation_weight*quaternion_loss

        # Obtain benchmark loss
        so3_target = R_target
        so3_estimate = quaternion_to_rotation_matrix(q_estimate)
        so3_distance = SO3_geodesic_distance(so3_estimate, so3_target)
        
        sequence_quaternion_loss += quaternion_loss
        sequence_t_loss += t_loss
        sequence_pose_loss += pose_loss
        benchmark_rotation_loss += so3_distance

    loss_dict["loss"] = sequence_pose_loss/seq_len
    loss_dict["rotation_loss"] = sequence_quaternion_loss/seq_len
    loss_dict["traslation_loss"] = sequence_t_loss/seq_len
    loss_dict["benchmark_rotation_loss"] = benchmark_rotation_loss/seq_len

    return loss_dict


def quaternion_pose_loss(data_dict,t_weight = 1, orientation_weight = 14.):
    ''' Loss function for pose expressed as translation vector and quaternion. Note:
    data_dict["result"] is a tensor with shape (batch x sequence_len x 7)
    data_dict["poses"] is a list with len = sequence_len, each element in list
    is a tensor with shape (batch x (4x4))
    returns: average for batch and sequence of the loss'''
    estimate = data_dict["result"] # 1,1,7 - 1,3,7
    target = data_dict["poses"] # list of seq with (batch, 4x4) 1(1x(4,4)) - 3(1x(4,4))

    seq_len = estimate.size()[1] # (batch, seq, dim_pose)

    loss_dict = dict()
    sequence_pose_loss = 0.
    sequence_quaternion_loss = 0.
    sequence_t_loss = 0.
    benchmark_rotation_loss = 0.
    for i in range (0,seq_len): # relative T, we don take first frame in seq
        # preprocessing target values
        R_target = target[i][:, :3, :3]
        q_target = rotation_matrix_to_quaternion(R_target)
        t_target = target[i][:,:-1,-1]

        q_estimate = estimate[:, i, 3:]
        t_estimate = estimate[:, i, :3]

        quaternion_loss = quaternion_distance_metric(q_estimate,q_target)
        t_loss = vector_distance(t_target,t_estimate)
        pose_loss = t_weight*t_loss + orientation_weight*quaternion_loss

        # Obtain benchmark loss
        so3_target = R_target
        so3_estimate = quaternion_to_rotation_matrix(q_estimate)
        so3_distance = SO3_geodesic_distance(so3_estimate, so3_target)
        
        sequence_quaternion_loss += quaternion_loss
        sequence_t_loss += t_loss
        sequence_pose_loss += pose_loss
        benchmark_rotation_loss += so3_distance
    
    loss_dict["loss"] = sequence_pose_loss/seq_len
    loss_dict["rotation_loss"] = sequence_quaternion_loss/seq_len
    loss_dict["traslation_loss"] = sequence_t_loss/seq_len
    loss_dict["benchmark_rotation_loss"] = benchmark_rotation_loss/seq_len

    return loss_dict


def se3_chordal_loss(data_dict, t_weight = 1,orientation_weight = 153.):
    ''' Loss function for SE3. Note:
    data_dict["result"] is a tensor with shape (batch x sequence_len x 6)
    data_dict["poses"] is a list with len = sequence_len, each element in list
    is a tensor with shape (batch x (4x4))
    returns: average for batch and sequence of the loss'''
    estimate = data_dict["result"] # 1,1,6 - 1,3,6 
    target = data_dict["poses"] # list of seq with (batch, 4x4) 1(1x(4,4)) - 3(1x(4,4))
    
    seq_len = estimate.size()[1] # (batch, seq, dim_pose)
    
    loss_dict = dict()
    sequence_se3_loss = 0.
    sequence_rotation_loss = 0.
    sequence_pos_loss = 0.
    benchmark_rotation_loss = 0.
    for i in range (0,seq_len): # relative T, we don take first frame in seq
        # preprocessing target values
        T_kf_i_target = target[i]
        T_kf_i_estimate = se3_exp_map(estimate[:, i, :])

        se3_loss = SE3_chordal_metric(T_kf_i_estimate,T_kf_i_target, t_weight, orientation_weight) # already providing mean val
        so3_loss = SO3_chordal_metric(T_kf_i_estimate[:, :3, :3],T_kf_i_target[:, :3, :3])
        t_loss = vector_distance(T_kf_i_estimate[:,:-1,-1],T_kf_i_target[:,:-1,-1])

        so3_benchmark_distance = SO3_geodesic_distance(T_kf_i_estimate[:, :3, :3],T_kf_i_target[:, :3, :3])
        
        sequence_se3_loss += se3_loss
        sequence_rotation_loss += so3_loss
        sequence_pos_loss += t_loss
        benchmark_rotation_loss += so3_benchmark_distance

    loss_dict["loss"] = sequence_se3_loss/seq_len
    loss_dict["rotation_loss"] = sequence_rotation_loss/seq_len
    loss_dict["traslation_loss"] = sequence_pos_loss/seq_len
    loss_dict["benchmark_rotation_loss"] = benchmark_rotation_loss/seq_len
    return loss_dict


def mse_euler_pose_loss(data_dict,orientation_weight = 100.):
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
    benchmark_rotation_loss = 0.
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
   
        loss = orientation_weight * rot_loss + pos_loss
        loss_dict[f"loss_frame_{i}"] = loss

        so3_benchmark_distance = SO3_geodesic_distance(R_kf_i,euler_angles_to_matrix(euler_estimate,'XYZ'))

        sequence_loss += loss
        sequence_rotation_loss += rot_loss
        sequence_pos_loss += pos_loss
        benchmark_rotation_loss += so3_benchmark_distance

    loss_dict["loss"] = sequence_loss/seq_len
    loss_dict["rotation_loss"] = sequence_rotation_loss/seq_len
    loss_dict["traslation_loss"] = sequence_pos_loss/seq_len
    loss_dict["benchmark_rotation_loss"] = benchmark_rotation_loss/seq_len

    return loss_dict