import unittest
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader.data_loaders import SingleDataset, MultiDataset
from torch.utils.data import Dataset, DataLoader

def plot_route(gt, c_gt='g'):    

    gt = torch.stack(gt).view(len(gt),4,4).cpu().detach().numpy() 

    x = gt[:][:,0,3]  
    y = gt[:][:,1,3]
    z = gt[:][:,2,3]
   
    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax1 = fig.add_subplot(231)
    ax1.plot(x, y)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax2 = fig.add_subplot(234)
    ax2.plot(x, z)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax3 = fig.add_subplot(232)
    ax3.plot(y, z)
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")

class TestDatabaseDataloader(unittest.TestCase):
    @classmethod
    # @unittest.skip("Skipping TUM test")
    def test_SingleDataLoader_TUM(self):
        test_sequence = "rgbd_dataset_freiburg1_xyz"
        cfg_dir=os.path.join(os.getcwd(),"configs","data_loader","TUM", test_sequence, test_sequence+".yml")
        _dset = SingleDataset(cfg_dir)

        i=0
        T_target_prev = list()  

        for d in tqdm(_dset):
            if d["poses"] is not None:

                plt.imshow( d["keyframe"].permute(1, 2, 0)+.5)

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
    @unittest.skip("Skipping dataloader euroc test")
    def test_SingleDataLoader_euroc(self):
        test_sequence="MH_04_difficult"
        cfg_dir=os.path.join(os.getcwd(),"configs","data_loader","EuRoC", test_sequence, test_sequence+".yml")
        _dset = SingleDataset(cfg_dir)
        ini = torch.Tensor([[0.27455972872811074, 0.7757302263680196, 0.5682073312266986, 4.677066],
                            [0.3078646952508953, -0.630726914634179, 0.7123221803187954, -1.74944],
                            [0.9109535030827971, -0.020644007727912694, -0.4119921603211767, 0.568567],
                            [0.,0.,0.,1.]])

        i=0
        T_target_prev = list()  
        for d in tqdm(_dset):
            # plt.imshow( d["keyframe"].permute(1, 2, 0)+.5)
            if d["poses"] is not None:
                H_kf0_kf1 = d["poses"][0]
                if i ==0:
                    T_target_prev.append(torch.matmul(H_kf0_kf1,ini))
                    i += 1 
                    continue # in first it go to next frame to retrieve rel pose
                # Absolute values
                H_0_kf0 = T_target_prev[-1]
                H_0_kf1 =torch.matmul(H_0_kf0,H_kf0_kf1)
                
                T_target_prev.append(H_0_kf1)
                
                i += 1 

                # plt.pause(0.005)

        # plt.show(block=False)
        # plt.pause(.003)
        # plt.close()

        plot_route(T_target_prev, c_gt='g')
        plt.show()
        plt.show(block=False) # uncomment if you want it to auto close
        plt.pause(3)
        plt.close()

    @classmethod
    @unittest.skip("Skipping dataloader Aqualoc test")
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
    @unittest.skip("Skipping dataloader MIMIR test")
    def test_SingleDataLoader_MIMIR(self):
        test_sequence="SeaFloor/track1"
        cfg_dir=os.path.join(os.getcwd(),"configs","data_loader","MIMIR", test_sequence+".yml")
        _dset = DataLoader(SingleDataset(cfg_dir),batch_size=1, shuffle=False, num_workers=0, drop_last=True)

        i=0
        T_target_prev = list()  

        for index,d in tqdm(enumerate(_dset), total=len(_dset)):
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

        plt.show(block=False)
        plt.pause(.003)
        plt.close()

        plot_route(T_target_prev, c_gt='g')
        plt.show()
        plt.show(block=False) # uncomment if you want it to auto close
        plt.pause(3)
        plt.close()


    @classmethod
    @unittest.skip("Skipping dataloader MIMIR test")
    def test_MultiDataLoader_MIMIR(self):
        test_sequences=["SeaFloor/track0", "SeaFloor/track1"]
        cfg_dirs = [os.path.join(os.getcwd(),"configs","data_loader","MIMIR", test_sequence+".yml") for test_sequence in test_sequences]
        # _dsets = [Dataset(cfg_dir) for cfg_dir in cfg_dirs]
        _dset = DataLoader(MultiDataset(cfg_dirs),batch_size=1, shuffle=False, num_workers=0, drop_last=True)

        i=0
        T_target_prev = list()  

        for index,d in tqdm(enumerate(_dset), total=len(_dset)):
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

        plt.show(block=False)
        plt.pause(.003)
        plt.close()

        plot_route(T_target_prev, c_gt='g')
        plt.show()
        plt.show(block=False) # uncomment if you want it to auto close
        plt.pause(3)
        plt.close()

    @classmethod
    @unittest.skip("Skipping dataloader KITTI test")
    def test_MultiDataLoader_KITTI(self):
        test_sequences=["05"]
        cfg_dirs = [os.path.join(os.getcwd(),"configs","data_loader","KITTI",test_sequence, test_sequence+".yml") for test_sequence in test_sequences]
        # _dsets = [Dataset(cfg_dir) for cfg_dir in cfg_dirs]
        _dset = DataLoader(MultiDataset(cfg_dirs),batch_size=1, shuffle=False, num_workers=0, drop_last=True)

        i=0
        T_target_prev = list()  

        for index,d in tqdm(enumerate(_dset), total=len(_dset)):
            # plt.imshow( d["keyframe"][0].permute(1, 2, 0)+.5)
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

        plt.show(block=False)
        plt.pause(.003)
        plt.close()

        plot_route(T_target_prev, c_gt='g')
        plt.show()
        plt.show(block=False) # uncomment if you want it to auto close
        plt.pause(3)
        plt.close()


if __name__ == '__main__':
    unittest.main()