from torch.utils.data import Dataset
from utils.loadconfig import ConfigLoader
from utils.general import mkdir_if_not_exists
import cv2
import torch

from .kitti import KittiOdom, KittiRaw
from .tum import TUM
from .euroc import EUROC
from .tartanair import TartanAIR
from .mimir import MIMIR
from .aqualoc import AQUALOC

datasets = {
            "kitti_odom": KittiOdom,
            "kitti_raw": KittiRaw,
            "tum-1": TUM,
            "tum-2": TUM,
            "tum-3": TUM,
            "euroc": EUROC,
            "tartanair": TartanAIR,
            "mimir": MIMIR,
            "aqualoc": AQUALOC,
        }


class MultiDataset(Dataset):
    """Class to load multiple sequneces as single dataset"""

    def __init__(self, dataset_dirs, **kwargs):
        if isinstance(dataset_dirs, list):
            self.datasets = [SingleDataset(dataset_dir, **kwargs) for dataset_dir in dataset_dirs]
        else:
            self.datasets = [SingleDataset(dataset_dirs, **kwargs)]
    

    def __getitem__(self, index):
        """
        returns a frame from a data set using flat index
        """
        for dataset in self.datasets:
            l = len(dataset)
            if index >= l:
                index -= l
            else:
                return dataset.__getitem__(index)
        return None

    def __len__(self):
        sum = 0
        for dataset in self.datasets:
            sum += len(dataset)
        return sum
class SingleDataset(Dataset):
    """ loads a single sequence as data set"""    
    def __init__(self, cfg):        
        # Read config file
        config_loader = ConfigLoader()
        self.cfg = self.read_cfg(config_loader,cfg)
        self.dataset = datasets[self.cfg.dataset](self.cfg)

    def read_cfg(self, config_loader, cfg_dir):
        cfg = config_loader.merge_cfg(cfg_dir)
        cfg.seq = str(cfg.seq)
        ''' double check result directory '''
        mkdir_if_not_exists(cfg.directory.result_dir)
        return cfg

    def preprocess_image(self,img,intrinsics,distcoeffs=None,crop_box=None):
        # undistort image
        if distcoeffs is not None:
            img = cv2.undistort(img, intrinsics, distcoeffs)
        # crop image
        if crop_box is not None:
            img = img[crop_box[1]:crop_box[1]+crop_box[3], crop_box[0]:crop_box[0]+crop_box[2]]
        # resize image
        if self.cfg.resize is not None:
            img = cv2.resize(img, (self.cfg.resize[1], self.cfg.resize[0]))
        # convert to grayscale
        if self.cfg.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # convert to tensor
        # img = torch.from_numpy(img).float()
        image_tensor = torch.tensor(img).to(dtype=torch.float32)
        image_tensor = image_tensor / 255 - .5
        if len(image_tensor.shape) == 2:
            image_tensor = torch.stack((image_tensor, image_tensor, image_tensor))
        else:
            image_tensor = image_tensor.permute(2, 0, 1)
        return image_tensor

    def __getitem__(self, index: int):

        # keyframe
        keyframe_id = index
        keyframe_timestamp = self.dataset.get_timestamp(index)
        keyframe = self.dataset.get_image(keyframe_timestamp)
        intrinsics = self.dataset.get_intrinsics_param()
        keyframe = self.preprocess_image(keyframe, intrinsics,self.cfg.crop_box)
        keyframe_abs_gt = self.dataset.get_gt_poses()[keyframe_timestamp]

        # frames of sequence
        frame_indexes = [i+index+1 for i in range(0, self.cfg.seq_len)]
        frame_timestamps = [self.dataset.get_timestamp(i) for i in frame_indexes]
        frames = [self.preprocess_image(self.dataset.get_image(i),intrinsics,self.cfg.crop_box) for i in frame_timestamps]
        frame_abs_gts = [self.dataset.get_gt_poses()[i] for i in frame_timestamps]

        data = {
            "keyframe": keyframe, # the reference image
            "keyframe_pose": torch.eye(4, dtype=torch.float32), # always identity
            "keyframe_intrinsics": intrinsics,
            "frames": [self.dataset.get_image(i) for i in frame_timestamps], # (ordered) neighboring images
            # "poses": poses, # H_ref_src
            "intrinsics": intrinsics,
            "image_id": index
        }

        return data

    def __len__(self) -> int:
        return len(self.dataset)
        