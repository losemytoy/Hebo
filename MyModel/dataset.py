import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import cv2
import tifffile as tif
import albumentations as A


from MyModel.until import transforms


class Segmentation(data.Dataset):
    def __init__(self, images, image_folder, mask_folder, train_val='train'):
        super(Segmentation, self).__init__()
        self.images = images
        self.image_folder = image_folder
        self.mask_folder = mask_folder

        if train_val == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize(256, 256),
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomCrop(220),
                transforms.ConvertArray(),
                transforms.ATransform(),
                transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(256, 256),
                transforms.ConvertArray(),
                transforms.ToTensor(),
            ])

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.images[idx])
        mask_path = os.path.join(self.mask_folder, self.images[idx])
        image = tif.imread(image_path)
        mask = tif.imread(mask_path)
        image, mask = self.transforms(image=image, target=mask)
        return image, mask

    def __len__(self):
        return len(self.images)
