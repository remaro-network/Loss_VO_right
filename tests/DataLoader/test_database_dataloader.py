import unittest
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader.data_loaders import SingleDataset

def plot_route(gt, c_gt='g'):    

    gt = torch.stack(gt).view(len(gt),4,4).cpu().detach().numpy() 

    x = gt[:][:,0,3]  
    y = gt[:][:,1,3]

    plt.plot(x, y, color=c_gt, label='Ground Truth')
    plt.scatter(x, y, color='b')
    plt.gca().set_aspect('equal', adjustable='datalim')

class TestDatabaseDataloader(unittest.TestCase):
    @classmethod
    # @unittest.skip("Skipping TUM test")
    def test_SingleDataLoader_TUM(self):
        test_sequence = "rgbd_dataset_freiburg1_360"
        cfg_dir=os.path.join(os.getcwd(),"configs","data_loader","TUM", test_sequence, test_sequence+".yml")
        _dset = SingleDataset(cfg_dir)

        i=0
        T_target_prev = list()  

        for d in tqdm(_dset):
            if d["poses"] is not None:

                # plt.imshow( d["keyframe"].permute(1, 2, 0)+.5)

                H_kf0_kf1 = d["poses"][0]
                if i ==0:
                    T_target_prev.append(H_kf0_kf1)
                    i += 1 
                    continue # in first it go to next frame to retrieve rel pose
                # Absolute values
                H_0_kf0 = T_target_prev[-1]
                H_0_kf1 =torch.matmul(H_0_kf0,H_kf0_kf1)
                
                T_target_prev.append(H_0_kf1)
                
                i += 1 

                plt.pause(0.005)


        # plt.show(block=False)
        # plt.pause(.003)
        # plt.close()

        plot_route(T_target_prev, c_gt='g')
        plt.show()
        plt.show(block=False) # uncomment if you want it to auto close
        plt.pause(3)
        plt.close()

    @classmethod
    # @unittest.skip("Skipping dataloader euroc test")
    def test_SingleDataLoader_euroc(self):
        test_sequence="MH_04_difficult"
        cfg_dir=os.path.join(os.getcwd(),"configs","data_loader","EuRoC", test_sequence, test_sequence+".yml")
        _dset = SingleDataset(cfg_dir)

        i=0
        T_target_prev = list()  

        for d in tqdm(_dset):
            # plt.imshow( d["keyframe"].permute(1, 2, 0)+.5)
            if d["poses"] is not None:
                H_kf0_kf1 = d["poses"][0]
                if i ==0:
                    T_target_prev.append(H_kf0_kf1)
                    i += 1 
                    continue # in first it go to next frame to retrieve rel pose
                # Absolute values
                H_0_kf0 = T_target_prev[-1]
                H_0_kf1 =torch.matmul(H_0_kf0,H_kf0_kf1)
                
                T_target_prev.append(H_0_kf1)
                
                i += 1 

                plt.pause(0.005)

        # plt.show(block=False)
        # plt.pause(.003)
        # plt.close()

        plot_route(T_target_prev, c_gt='g')
        plt.show()
        plt.show(block=False) # uncomment if you want it to auto close
        plt.pause(3)
        plt.close()

    @classmethod
    # @unittest.skip("Skipping dataloader Aqualoc test")
    def test_SingleDataLoader_aqualoc(self):
        test_sequence="1"
        cfg_dir=os.path.join(os.getcwd(),"configs","data_loader","Aqualoc/Archaeological_site_sequences", test_sequence, test_sequence+".yml")
        _dset = SingleDataset(cfg_dir)

        i=0
        T_target_prev = list()  

        for d in tqdm(_dset):
            # plt.imshow( d["keyframe"].permute(1, 2, 0)+.5)
            if d["poses"] is not None:
                H_kf0_kf1 = d["poses"][0]
                if i ==0:
                    T_target_prev.append(H_kf0_kf1)
                    i += 1 
                    continue # in first it go to next frame to retrieve rel pose
                # Absolute values
                H_0_kf0 = T_target_prev[-1]
                H_0_kf1 =torch.matmul(H_0_kf0,H_kf0_kf1)
                
                T_target_prev.append(H_0_kf1)
                
                i += 1 

                plt.pause(0.005)

        # plt.show(block=False)
        # plt.pause(.003)
        # plt.close()

        plot_route(T_target_prev, c_gt='g')
        plt.show()
        plt.show(block=False) # uncomment if you want it to auto close
        plt.pause(3)
        plt.close()

    @classmethod
    # @unittest.skip("Skipping dataloader MIMIR test")
    def test_SingleDataLoader_MIMIR(self):
        test_sequence="SeaFloor/track0"
        cfg_dir=os.path.join(os.getcwd(),"configs","data_loader","MIMIR", test_sequence+".yml")
        _dset = SingleDataset(cfg_dir)

        i=0
        T_target_prev = list()  

        for d in tqdm(_dset):
            # plt.imshow( d["keyframe"].permute(1, 2, 0)+.5)
            if d["poses"] is not None:
                H_kf0_kf1 = d["poses"][0]
                if i ==0:
                    T_target_prev.append(H_kf0_kf1)
                    i += 1 
                    continue # in first it go to next frame to retrieve rel pose
                # Absolute values
                H_0_kf0 = T_target_prev[-1]
                H_0_kf1 =torch.matmul(H_0_kf0,H_kf0_kf1)
                
                T_target_prev.append(H_0_kf1)
                
                i += 1 

                plt.pause(0.005)

        # plt.show(block=False)
        # plt.pause(.003)
        # plt.close()

        plot_route(T_target_prev, c_gt='g')
        plt.show()
        plt.show(block=False) # uncomment if you want it to auto close
        plt.pause(3)
        plt.close()



if __name__ == '__main__':
    unittest.main()