import unittest
import torch
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader.data_loaders import SingleDataset
class TestDatabaseDataloader(unittest.TestCase):
    @classmethod
    # def setUpClass(cls):
    #     cls._db = SfmExporter()
    def test_SingleDataLoader(self):
        cfg_dir=os.path.join(os.getcwd(),"configs","data_loader","EuRoC","MH_04_difficult.yml")
        _dset = SingleDataset(cfg_dir)

        for d in tqdm(_dset):
            plt.imshow( d["keyframe"].permute(1, 2, 0)+.5)
            plt.pause(0.05)
        plt.show()
        cv2.destroyAllWindows()




if __name__ == '__main__':

    unittest.main()

