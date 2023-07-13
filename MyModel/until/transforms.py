import numpy as np
import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from skimage.transform import resize
import albumentations as A


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, [0, 0, padw, padh], fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 将image 和 target 的短边缩放到size大小
        image = F.resize(image, [size])
        target = F.resize(target, [size], interpolation=transforms.InterpolationMode.NEAREST)
        return image, target


class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, target):
        # resize = transforms.Resize([self.height, self.width])
        # image = resize(image)
        # target = resize(target)
        image = resize(image, (self.height, self.width, 5))
        target = resize(target, (self.height, self.width))
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = A.Flip(image)
            target = A.Flip(target)
            # image = F.hflip(image)
            # target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = transforms.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        # image = F.to_tensor(image)
        # target = torch.as_tensor(np.array(target), dtype=torch.int64)

        image = torch.Tensor(image)
        target = np.expand_dims(target, axis=0)
        target = target / 255.0
        target = torch.Tensor(target)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ConvertArray(object):
    def __call__(self, image, target):
        image, target = np.array(image), np.array(target)
        image = np.transpose(image, (2, 0, 1))
        first_four_layers = image[:4]
        last_layer = image[4]
        first_four_layers = first_four_layers / 65535.0
        result = np.concatenate((first_four_layers, last_layer[np.newaxis]), axis=0)
        target = np.expand_dims(target, axis=0)
        target = target / 255.0
        return image, target


class ATransform(object):
    def __init__(self):
        self.t = A.Compose([
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.2),
        ])

    def __call__(self, image, target):
        augmented = self.t(image=image)
        return augmented['image'], target
