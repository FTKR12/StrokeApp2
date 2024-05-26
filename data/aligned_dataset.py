import os
import random
import torchvision.transforms as transforms
import torch.utils.data as data

from data.image_folder import make_dataset
from PIL import Image

from data.base_dataset import BaseDataset

import torch
import torchvision.transforms.functional as F

class RandomCrop(torch.nn.Module):

    @staticmethod
    def get_params(ct, output_size):
        w, h = ct.size
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw


    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = size

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, ct, mri):
        
        width, height = ct.size
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            ct = F.pad(ct, padding, self.fill, self.padding_mode)
            mri = F.pad(mri, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            ct = F.pad(ct, padding, self.fill, self.padding_mode)
            mri = F.pad(mri, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(ct, self.size)

        return F.resized_crop(ct, i, j, h, w, (width, height)), F.resized_crop(mri, i, j, h, w, (width, height))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"


class RandomHorizontalFlip(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, ct, mri, mask):
        if torch.rand(1) < self.p:
            return F.hflip(ct), F.hflip(mri), F.hflip(mask)
        return ct, mri, mask


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

class RandomVerticalFlip(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, ct, mri, mask):
        if torch.rand(1) < self.p:
            return F.vflip(ct), F.vflip(mri), F.vflip(mask)
        return ct, mri, mask


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.dir_mask = os.path.join(opt.maskroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB))  # get image paths
        self.mask_paths = sorted(make_dataset(self.dir_mask))  # get image paths
        #assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        AB_path = self.AB_paths[index]
        mask_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('I;16')
        mask = Image.open(mask_path).convert('L')
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h)).resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
        B = AB.crop((w2, 0, w, h)).resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
        
        # aug
        if self.opt.phase == 'train':
            A, B, mask = RandomHorizontalFlip(0.5)(A, B, mask)
            A, B, mask = RandomVerticalFlip(0.5)(A, B, mask)
        
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        mask = transforms.ToTensor()(mask)

        A = A / 5000
        B = B / 5000

        w_offset = random.randint(0, max(0, self.opt.load_size - self.opt.load_size - 1))
        h_offset = random.randint(0, max(0, self.opt.load_size - self.opt.load_size - 1))
        A = A[:, h_offset:h_offset + self.opt.load_size, w_offset:w_offset + self.opt.load_size]
        B = B[:, h_offset:h_offset + self.opt.load_size, w_offset:w_offset + self.opt.load_size]
        mask = mask[:, h_offset:h_offset + self.opt.load_size, w_offset:w_offset + self.opt.load_size]

        A = transforms.Normalize((0.5), (0.5))(A)
        B = transforms.Normalize((0.5), (0.5))(B)

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'mask': mask, 'mask_path': mask_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
