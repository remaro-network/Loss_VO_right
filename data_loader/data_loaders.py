from torch.utils.data import Dataset
from utils.loadconfig import ConfigLoader
from utils.general import mkdir_if_not_exists
import cv2
import torch
import numpy as np
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
        self.gt_timestamps = []

        self.dataset = datasets[self.cfg.dataset](self.cfg)
        # read camera intrinsics
        self.intrinsics = self.dataset.get_intrinsics_param()
        self.distortion_params = self.dataset.get_distortion_param()
        # synchronize timestamps
        self.dataset.synchronize_timestamps()
        # get gt poses
        self.gt_poses = self.dataset.get_gt_poses()
        self.dataset.update_gt_pose()
        

    def read_cfg(self, config_loader, cfg_dir):
        cfg = config_loader.merge_cfg(cfg_dir)
        cfg.seq = str(cfg.seq)
        ''' double check result directory '''
        mkdir_if_not_exists(cfg.directory.result_dir)
        return cfg

    def preprocess_image(self,img,intrinsics,distcoeffs=None,crop_box=None):
        ''' Undistort, and/or crop and/or resize image
        Args:
            img: image to be processed
            intrinsics: camera intrinsics
            distcoeffs: distortion coefficients
            crop_box: crop box [ow,oh,bw,bh] or ['random','random',bw,h]
        Returns:
            img: processed image
            intrinsics: processed camera intrinsics

        The image is cropped as follows:
            0                    w
         0  +--------------------+->
            |    ow          bw  |
            | oh +-----------+   |
            |    |           |   |
            |    |   image   |   |
            |    |           |   |
            | bh +-----------+   |
            |                    |
         h  +--------------------+
            |
            v
        '''
        # undistort image
        h,w,ch = img.shape
        img = img.astype(np.float32)

        if distcoeffs is not None:
            distcoeffs = np.asarray(distcoeffs)

            intrinsics = np.array([[intrinsics[0],0,intrinsics[2]],[0,intrinsics[1],intrinsics[3]],[0,0,1]]).astype(np.float32)
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsics,distcoeffs,(w,h),1,(w,h))
            # mapx,mapy = cv2.fisheye.initUndistortRectifyMap(intrinsics,distcoeffs,np.eye(3),intrinsics,(w,h),cv2.CV_32F)
            mapx,mapy = cv2.initUndistortRectifyMap(intrinsics,distcoeffs,np.eye(3),newcameramtx,(w,h),cv2.CV_32F)
            img = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

        # crop image
        if crop_box is not None:
            img = img[crop_box[1]:crop_box[1]+crop_box[3], crop_box[0]:crop_box[0]+crop_box[2]]
            cy = img.shape[0]/2
            cx = img.shape[1]/2
            intrinsics[0][2] = cx
            intrinsics[1][2] = cy
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
        return image_tensor,intrinsics

    def __getitem__(self, index: int):
        if index == len(self.dataset.rgb_d_pose_pair):
            raise IndexError
        if self.cfg.crop_box is not None:
            crop_box = self.cfg.crop_box.copy()
        # setup crop box if random
            if crop_box[0]=='random':
                crop_box[0] = np.random.randint(0,int(self.cfg.image.width)-crop_box[2])
                print('crop_box[0]',crop_box[0])
            if crop_box[1]=='random':
                crop_box[1] = np.random.randint(0,int(self.cfg.image.height)-crop_box[3])
                print('crop_box[1]',crop_box[1])
        else:
            crop_box = None
        # keyframe
        keyframe_id = index
        keyframe_timestamp = self.dataset.get_timestamp(index)
        keyframe = self.dataset.get_image(keyframe_timestamp)
        keyframe, kf_intrinsics = self.preprocess_image(keyframe, self.intrinsics,self.distortion_params,crop_box)

        # frames of sequence
        frame_indexes = [i+index+1 for i in range(0, self.cfg.seq_len)]
        
        if any(i >= len(self.dataset.rgb_d_pose_pair) for i in frame_indexes):
            frames = None
            frame_rel_gts = None
        else:
            keyframe_abs_gt = torch.from_numpy(self.dataset.gt_poses[index]).to(dtype=torch.float32) 
            frame_timestamps = [self.dataset.get_timestamp(i) for i in frame_indexes]
            frames, frame_intrinsics = zip(*[self.preprocess_image(self.dataset.get_image(i),self.intrinsics,self.distortion_params,crop_box) for i in frame_timestamps])
            frame_abs_gts = [torch.from_numpy(self.dataset.gt_poses[i]).to(dtype=torch.float32) for i in frame_indexes]
            frame_rel_gts = [torch.matmul(torch.inverse(keyframe_abs_gt), frame_abs_gt) for frame_abs_gt in frame_abs_gts]
        data = {
            "keyframe": keyframe, # the reference image
            "keyframe_pose": torch.eye(4, dtype=torch.float32), # always identity
            "keyframe_intrinsics": torch.tensor(kf_intrinsics).to(dtype=torch.float32),
            "frames": frames, # (ordered) neighboring images
            "poses": frame_rel_gts, # H_ref_src
            "intrinsics": torch.tensor(self.intrinsics).to(dtype=torch.float32),
            "image_id": torch.tensor(index)
        }
        if None in data.values():
            raise IndexError

        return data


    def __len__(self) -> int:
        return len(self.dataset.rgb_d_pose_pair)-self.cfg.seq_len
        