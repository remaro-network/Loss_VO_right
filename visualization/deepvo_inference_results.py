import os
import pathlib
import csv
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.conversions import angle_axis_to_rotation_matrix
import numpy as np
import matplotlib.pyplot as plt
from data_loader.data_loaders import SingleDataset
from model.deepvo.deepvo_model import DeepVOModel
from utils.conversions import se3_exp_map, quaternion_to_rotation_matrix

def plot_route(trajectories = None, labels = None, colors = None):
    '''Plots the trajectory of the robot in 3D space
    Args:
        trajectories (list): list of trajectories to plot
        labels (list): list of labels for each trajectory,
        colors (list): list of colors for each trajectory
        '''    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(234)
    ax3 = fig.add_subplot(232)
    for trajectory, label, color in zip(trajectories, labels, colors):
        trajectory = torch.stack(trajectory).view(len(trajectory),4,4).cpu().detach().numpy()
        x = trajectory[:][:,0,3]  
        y = trajectory[:][:,1,3]
        z = trajectory[:][:,2,3]
        a1 = ax1.plot(x, y, color=color, label=label)
        plt.xlabel("X")
        plt.ylabel("Y")
        a2 = ax2.plot(x, z, color=color, label=label)
        plt.xlabel("X")
        plt.ylabel("Z")
        a3 = ax3.plot(y, z, color=color, label=label)
        plt.xlabel("Y")
        plt.ylabel("Z")
    fig.legend([a1, a2, a3],     # The line objects
           labels=labels,   # The labels for each line
           loc="lower right",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Legend Title"  # Title for the legend
           )
    plt.show()

def save_trajectory_to_csv(trajectory= None, file_name = None):
    '''Saves the trajectory of the robot in 3D space
    Args:
        trajectories (list): list of trajectories to plot
        '''    
    trajectory = torch.stack(trajectory).view(len(trajectory),4,4).cpu().detach().numpy()
    x = trajectory[:][:,0,3]  
    y = trajectory[:][:,1,3]
    z = trajectory[:][:,2,3]
    with open(os.path.join(os.getcwd(),"visualization",file_name), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z"])
        for i in range(len(x)):
            writer.writerow([x[i], y[i], z[i]])

def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(data[k], device) for k in data.keys()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    else:
        return data.to(device)

def relative_to_absolute_pose(poses):
    '''Converts relative poses to absolute poses
    Args:
        poses (list): list of relative poses
    Returns:
        list: list of absolute poses
    '''
    absolute_poses = []
    for i,relative_pose in enumerate(poses):
        if i==0:
            absolute_poses.append(relative_pose)
            continue
        last_absolute_pose = absolute_poses[-1]
        absolute_poses.append(torch.matmul(last_absolute_pose, relative_pose))
    return absolute_poses

def main():
    # Load the data
    test_sequence='07'
    models = ['deepvo_original', 'deepvo_se3', 'deepvo_quat']
    # models = ['deepvo_original', 'deepvo_se3']
    cfg_dir=os.path.join(os.getcwd(),"configs","data_loader","KITTI", test_sequence, test_sequence+".yml")
    data_loader = DataLoader(SingleDataset(cfg_dir),batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    # Some auxiliary variables
    T_target_relative = list()  

    T_deepvo_relative = list()
    T_deepvo_se3_relative = list()
    T_deepvo_quat_relative = list()
    if 'deepvo_original' in models:
        deepvo_model = DeepVOModel(batchNorm = True, checkpoint_location=[os.path.join(os.getcwd(),"saved/deepvo/icra23", "best-checkpoint-epoch.pth")],
                            conv_dropout = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5],output_shape = 6, 
                            image_size = (371,1241), rnn_hidden_size=1000, rnn_dropout_out=.5, rnn_dropout_between=0).to(device)
        # Now let's run the model on the data   
        with torch.no_grad():     
            for batch_idx, data in tqdm(enumerate(data_loader),total=len(data_loader)):
                        # Every data instance is a pair of input data + target result
                        # data = to_device(data, "cuda:0" if torch.cuda.is_available() else "cpu")
                        data = to_device(data,"cuda:0" if torch.cuda.is_available() else "cpu")
                        # Make predictions for this batch
                        deepvo_outputs = deepvo_model(data)

                        target = data["poses"]
                        deepvo_estimate = deepvo_outputs["result"]

                        # Relative values
                        T_target_relative.append(target[0])
                        # for deepvo (w. angle axis rotations)
                        T_estimate_rel = angle_axis_to_rotation_matrix(deepvo_estimate[:, 1, :3])            
                        T_estimate_rel[:,0:3,3] = deepvo_estimate[:,1,-3:]
                        T_deepvo_relative.append(T_estimate_rel)
                        

                
        T_target_absolute = relative_to_absolute_pose(T_target_relative)
        T_deepvo_absolute = relative_to_absolute_pose(T_deepvo_relative)
        del deepvo_model
        # save absolute trajectories to csv file
        save_trajectory_to_csv(T_target_absolute, test_sequence+"_target_absolute.csv")
        save_trajectory_to_csv(T_deepvo_absolute, test_sequence+"_deepvo_absolute.csv")
        plot_route(trajectories=[T_target_absolute, T_deepvo_absolute], 
            colors=['darkseagreen', 'tomato'], labels=['Ground Truth', 'DeepVO'])     

    if 'deepvo_se3' in models:
        deepvo_se3_model = DeepVOModel(batchNorm = True, checkpoint_location=[os.path.join(os.getcwd(),"saved/deepvo_se3/icra23/2", "best-checkpoint-epoch.pth")],
                            conv_dropout = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5],output_shape = 6, 
                            image_size = (371,1241), rnn_hidden_size=1000, rnn_dropout_out=.5, rnn_dropout_between=0).to(device)
        # Now let's run the model on the data   
        with torch.no_grad():     
            for batch_idx, data in tqdm(enumerate(data_loader),total=len(data_loader)):
                        # Every data instance is a pair of input data + target result
                        data = to_device(data,"cuda:0" if torch.cuda.is_available() else "cpu")
                        # Make predictions for this batch

                        deepvo_se3_outputs = deepvo_se3_model(data)

                        target = data["poses"]
                        deepvo_se3_estimate = deepvo_se3_outputs["result"]

                        # Relative values
                        T_target_relative.append(target[0])
                        # for deepvo (w. se3 rotations)
                        T_estimate_se3_rel = se3_exp_map(deepvo_se3_estimate[:, 0, :6])
                        T_deepvo_se3_relative.append(T_estimate_se3_rel)

                
        T_deepvo_se3_absolute = relative_to_absolute_pose(T_deepvo_se3_relative)
        # del deepvo_se3_model
        # save absolute trajectories to csv file
        save_trajectory_to_csv(T_deepvo_se3_absolute, test_sequence+"_deepvo_se3_absolute.csv")
        plot_route(trajectories=[T_target_absolute, T_deepvo_se3_absolute], 
            colors=['darkseagreen', 'tomato'], labels=['Ground Truth', 'DeepVO (SE3)'])     

    if 'deepvo_quat' in models:
        deepvo_quat_model = DeepVOModel(batchNorm = True, checkpoint_location=[os.path.join(os.getcwd(),"saved/deepvo_quat/icra23", "best-checkpoint-epoch.pth")],                  
                            conv_dropout = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5],output_shape = 7, 
                            image_size = (371,1241), rnn_hidden_size=1000, rnn_dropout_out=.5, rnn_dropout_between=0).to(device)
        # Now let's run the model on the data   
        with torch.no_grad():     
            for batch_idx, data in tqdm(enumerate(data_loader),total=len(data_loader)):
                        # Every data instance is a pair of input data + target result
                        data = to_device(data,"cuda:0" if torch.cuda.is_available() else "cpu")
                        # Make predictions for this batch
                        deepvo_outputs = deepvo_quat_model(data)
                        deepvo_se3_outputs = deepvo_quat_model(data)
                        deepvo_quat_outputs = deepvo_quat_model(data)

                        target = data["poses"]
                        deepvo_quat_estimate = deepvo_quat_outputs["result"]

                        # Relative values
                        T_target_relative.append(target[0])
                    
                        # for deepvo (w. quaternion rotations)
                        T_estimate_quat_rel = torch.eye((4)).reshape(1,4,4)
                        q_rotation = quaternion_to_rotation_matrix(deepvo_quat_estimate[:, 0, 3:])
                        T_estimate_quat_rel[:,:3,:3] = q_rotation
                        T_estimate_quat_rel[:,0:3,3] = deepvo_quat_estimate[:,0,:3]
                        T_deepvo_quat_relative.append(T_estimate_quat_rel)

                

        T_deepvo_quat_absolute = relative_to_absolute_pose(T_deepvo_quat_relative)
        # save absolute trajectories to csv file
        save_trajectory_to_csv(T_deepvo_quat_absolute, test_sequence+"_deepvo_quat_absolute.csv")
        plot_route(trajectories=[T_target_absolute, T_deepvo_quat_absolute], 
            colors=['darkseagreen', 'tomato'], labels=['Ground Truth', 'DeepVO (Quat)'])     


    # Now let's plot the results
    plot_route(trajectories=[T_target_absolute, T_deepvo_absolute, T_deepvo_se3_absolute, T_deepvo_quat_absolute], 
            colors=['darkseagreen', 'tomato', 'steelblue', 'darkslateblue'], labels=['Ground Truth', 'DeepVO', 'DeepVO (SE3)', 'DeepVO (Quat)'])           
        




if __name__ == '__main__':
        main()
