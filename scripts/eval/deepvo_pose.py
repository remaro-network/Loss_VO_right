import argparse
import pathlib
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.model as module_arch
from utils.parse_config import ConfigParser
from utils import to, DS_Wrapper, inf_loop
from utils.conversions import angle_axis_to_rotation_matrix

import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_route(gt, out, c_gt='g', c_out='r'):    
    x_idx = 3
    y_idx = 5
    gt = torch.stack(gt).view(len(gt),4,4).cpu().detach().numpy() 
    out = torch.stack(out).view(len(out),4,4).cpu().detach().numpy() 

    x = gt[:][:,0,3]  
    y = gt[:][:,1,3]

    plt.plot(x, y, color=c_gt, label='Ground Truth')
    plt.scatter(x, y, color='b')

    x = out[:][:,0,3]
    y = out[:][:,1,3]
    plt.plot(x, y, color=c_out, label='DeepVO')
    plt.scatter(x, y, color='b')
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.pause(0.05)
    



def main(config):
    logger = config.get_logger('test')
    output_dir = pathlib.Path(config.config.get("output_dir", "saved"))
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # define output files here
    
    # setup data loader
    data_loader = config.initialize('data_loader', module_data)
    # data_loader = inf_loop(data_loader)

    # base_path = data_args['dataset_dir']
    # sequence = data_args['sequences']
    model = config.initialize('arch', module_arch)
    # logger.info(model)

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    T_result_prev = list()
    T_target_prev = list()   
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            # imgcv = data["keyframe"].view(3,480,752).cpu().detach().numpy()
            # imgcv = (imgcv[0,:,:]+0.5)*255
            # imgcv = imgcv.astype(np.uint8)
            # print(np.amin(imgcv))
            # print(np.amax(imgcv))
            # print(imgcv.shape)
            # plt.imshow(imgcv)
            # plt.show(block=False)
            # plt.pause(.3)
            # plt.close()
            # if i == 0:
            #     data["hidden"] = [1]
            #     print(data)

            
            # Every data instance is a pair of input data + target result
            data = to(data, device)
            
            result = model(data)
            output = result["result"]
            target = data["poses"]
            # print('\n len',len(target))
            # print(target)

            # preprocessing result values
            # Relative values
            T_result_rel = angle_axis_to_rotation_matrix(output[:, 0, :3])

            T_result_rel[:,0:3,3] = output[:,0,-3:]
            H_kf0_kf1 = target[0]
            
            
        
            if i ==0:
                T_result_prev.append(H_kf0_kf1) # pose of first KF is initial pose
                T_target_prev.append(H_kf0_kf1)
                continue # in first it go to next frame to retrieve rel pose
            # Absolute values
            H_0_kf0 = T_target_prev[-1]
            H_0_kf1 =torch.matmul(H_0_kf0,H_kf0_kf1)
            
            T_result_prev.append(torch.matmul(T_result_rel[0],T_result_prev[-1]))
            T_target_prev.append(H_0_kf1)

            plot_route(T_target_prev, T_result_prev, c_gt='g', c_out='r')
        
        plt.show()


   

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)