import os
import glob
import torch
import torch.utils.data as data
import torchvision
import torchvision.io as io
from torchvision import transforms
import torchvision.transforms.functional as f
from PIL import Image
import random


def make_datapath_list(iorm='img', path='img', phase="train", rate=0.8):
    """
    make filepath list for train and validation image and mask.
    """
    rootpath = "./dataset/" + path
    target_path = os.path.join(rootpath + '/*.jpg')
    # print(target_path)

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    if phase == 'train' and iorm == 'img':
        num = len(path_list)
        random.shuffle(path_list)
        return path_list[:int(num * rate)], path_list[int(num * rate):]

    elif phase == 'test' or iorm == 'mask':
        return path_list


class ImageTransform:
    """
    preprocessing images
    """

    def __init__(self, size, mean, std):
        self.data_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean, std)])

    def __call__(self, img):
        mirror = f.hflip(img)
        img = torch.concat([img, mirror], dim=2)
        return self.data_transform(img)


'''
class MaskTransform:
    """
    preprocessing images
    """
    def __init__(self, size):
        self.data_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor()])

    def __call__(self, img):
        return self.data_transform(img)

'''


class ImageDataset(data.Dataset):
    """
    Dataset class. Inherit Dataset class from PyTorch.
    """

    def __init__(self, img_list, img_transform, mask_width=0.5):
        self.img_list = img_list
        self.img_transform = img_transform
        self.mask_width = mask_width
        assert 0 < mask_width < 1

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        """
        get tensor type preprocessed Image
        """
        gt = self.img_transform(io.read_image(self.img_list[index], mode=io.ImageReadMode.RGB))

        c = gt.size(dim=0)
        h = gt.size(dim=1)
        w = gt.size(dim=2)

        mask = torch.concat([torch.ones(c, h, round(w/2)),
                             torch.zeros(c, h, round(w/2 * self.mask_width)),
                             torch.ones(c, h, w - (round(w/2) + round(w/2 * self.mask_width)))
                             ],
                            dim=2)

        assert gt.size() == mask.size()

        return gt * mask, mask, gt
