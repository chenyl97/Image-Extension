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


def make_datapath_list(iorm='img', path='img', phase="train", rate=0.8, dataset_num=10000):
    """
    make filepath list for train and validation image and mask.
    """
    rootpath = "project/dataset/" + path
    classes = ['aquarium', 'athletic_ﬁeld/outdoor', 'beach', 'cliff', 'coast', 'forest_path', 'golf_course', 'harbor',
               'lake/natural', 'mountain', 'ocean', 'pier', 'pond', 'rainforest', 'river', 'skyscraper', 'swamp',
               'underwater/ocean_deep', 'valley', 'vegetable_garden']

    target_paths = []
    for cls in classes:
        cls = '/' + cls[0] + '/' + cls
        target_path = os.path.join(rootpath + cls + '/*.jpg')
        target_paths.append(target_path)

    path_list = []
    print(target_paths[0])

    for target_path in target_paths:
        for path in glob.glob(target_path):
            path_list.append(path)

    if phase == 'train' and iorm == 'img':
        random.shuffle(path_list)
        path_list = path_list[:dataset_num]
        return path_list[:int(dataset_num * rate)], path_list[int(dataset_num * rate):]

    elif phase == 'test' or iorm == 'mask':
        return path_list


class ImageTransform:
    """
    preprocessing images
    """

    def __init__(self, mean, std, size):
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

        # print('mask_size is {}'.format(mask.size()))
        # print('gt size is {}'.format(gt.size()))

        return gt * mask, mask, gt
