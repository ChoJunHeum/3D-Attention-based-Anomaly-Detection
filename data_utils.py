import os
import random
from functools import reduce

import glob
import torch
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406]).to('cuda')
STD = torch.tensor([0.229, 0.224, 0.225]).to('cuda')
MEAN4 = MEAN[None, :, None, None]
STD4 = STD[None, :, None, None]
MEAN5 = MEAN[None, :, None, None, None]
STD5 = STD[None, :, None, None, None]


def get_transform(cfg):
    return transforms.Compose([           ## https://pytorch.org/hub/pytorch_vision_vgg/
                transforms.Resize([cfg.resize_h, cfg.resize_w]),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN,
                                    std=STD),
                ])


def inverse_norm(img_t, inverse_transform=None):
    '''
    ! Not using now... Use just hard coded MEAN4, MEAN5, ...!
    [C, h, w] -> MEAN, STD to [:, None, None]
    [B, C, h, w] -> MEAN, STD to [None, :, None, None]
    [B, C, D, h, w] -> MEAN, STD to [None, :, None, None, None]
    
    Args:
        inverse_transform: if given, return PIL Image. 
            transforms.Compose([
                transforms.ToPILImage()
                transforms.Resize([original_h, original_w])
            ])
    '''
    img_shape = len(img_t.size())
    if img_shape > 3:
        mean = MEAN.reshape(MEAN.size() + (1,)*(img_shape - 2))[None,:].to(img_t.device)
        std = STD.reshape(STD.size() + (1,)*(img_shape - 2))[None,:].to(img_t.device)
    else:
        mean = MEAN.reshape(MEAN.size() + (1,)*(img_shape - 1)).to(img_t.device)
        std = STD.reshape(STD.size() + (1,)*(img_shape - 1)).to(img_t.device)
    img_t = img_t * std + mean
    if inverse_transform is not None:
        return inverse_transform(img_t)
    return img_t


def img2tensor(raw_img, transform, symmetric=False):
    """
    Load and resize img, convert to tensor then map to device.
    Image.open: [W, H]
    ToTensor: [C, H, W]
    if symmetric==True, [-1., 1]
    else, [0., 1.]
    Return tensor, shape: CHW (RGB)
    """
    if isinstance(raw_img, str):
        img = Image.open(raw_img)
    else:
        img = raw_img
    img_t = transform(img)
    if symmetric:
        img_t = img_t*2-1

    return img_t


class CropTrainDataset(Dataset):
    """
    cv2 imread: BGR, HWC (H, W, C)
    PIL open: RGB, HWC (H, W, C)
    ToTensor: [C, H, W]

    batches:
        x_t: [B, C_out * nframe, resize_h, resize_w]
        y_t: [B, C_out, resize_h, resize_w]
    """
    def __init__(self, cfg):
        self.resize_h = cfg.resize_h
        self.resize_w = cfg.resize_w
        self.symmetric = cfg.symmetric
        self.device = cfg.device
        self.transform = get_transform(cfg)
        total_frames = cfg.nframe + 1 # input_frame + target_frame

        self.all_path = [] # len(self.all_path) = nframe
        for folder_t in sorted(glob.glob(f'{cfg.tr_path}/*')): # t0, t1, ...
            path_infolder = glob.glob(f'{folder_t}/*.jpg')
            path_infolder.sort()
            self.all_path.append(path_infolder) 
        lengths = [len(folder_t) for folder_t in self.all_path]
        total_length = reduce(lambda x, y: x+y, lengths)
        assert len(self.all_path) == total_frames, f'{len(self.all_path)} and {total_frames}'
        assert total_length == lengths[0] * total_frames, f'{total_length} and {lengths[0] * total_frames}'

    def __len__(self): 
        '''
        Return length of paired sets
        '''
        return len(self.all_path[0])

    def __getitem__(self, index):
        '''
        frames_all = [ img_from_t0, img_from_t1, ... ]
        len(frames_all) == nframe
        one_path[i]: shape [3, resize_h, resize_w]
        '''
        one_patch_path = [folder_t[index] for folder_t in self.all_path]

        one_patch = [img2tensor(path,
                        transform=self.transform,
                        symmetric=self.symmetric,
                        ) for path in one_patch_path]
        
        # x_t = torch.cat(one_patch[:-1], dim=0).to(self.device)
        x_t = torch.stack(one_patch[:-1], dim=1).to(self.device)
        y_t = torch.as_tensor(one_patch[-1], device=self.device)

        return x_t, y_t


class TestDataset:
    def __init__(self, cfg, folder):
        self.total_frames = cfg.nframe + 1 # input_frame + target_frame
        self.imgs = glob.glob(f'{folder}/*.jpg')
        self.imgs.sort()

    def __len__(self):
        return len(self.imgs) - (self.total_frames - 1)

    def __getitem__(self, indice):
        frames = []
        for i in range(indice, indice + self.total_frames):
            frames.append(Image.open(self.imgs[i]))

        return frames[:-1], frames[-1]


class Label_loader:
    def __init__(self, cfg, folders):
        assert cfg.te_name in ('ped2', 'avenue', 'shanghaitech'), f'Did not find the related gt for \'{cfg.te_name}\'.'
        self.te_name = cfg.te_name
        # self.mat_path = f'{cfg.te_path}/{self.te_name}'
        self.mat_path = f'{cfg.te_path}'
        self.folders = folders

    def __call__(self):
        if self.te_name == 'shanghaitech':
            gt = self.load_shanghaitech()
        else:
            gt = self.load_ucsd_avenue()
        return gt

    def load_ucsd_avenue(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']

        all_gt = []
        for i in range(abnormal_events.shape[0]):
            length = len(os.listdir(self.folders[i]))
            sub_video_gt = np.zeros((length,), dtype=np.int8)
            one_abnormal = abnormal_events[i]

            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

            for j in range(one_abnormal.shape[1]):
                start = one_abnormal[0, j] - 1
                end = one_abnormal[1, j]
                sub_video_gt[start: end] = 1
            all_gt.append(sub_video_gt)

        return all_gt


    def load_shanghaitech(self):
        np_list = glob.glob(f'{self.mat_path}/testing/testframemask/*.npy')
        np_list.sort()

        gt = []
        for npy in np_list:
            gt.append(np.load(npy))

        return gt
