import os
import torch.utils.data as data
from PIL import Image

from models.until import transforms


class Segmentation(data.Dataset):
    def __init__(self, images, image_folder, mask_folder, train_val='train'):
        super(Segmentation, self).__init__()
        self.images = images
        self.image_folder = image_folder
        self.mask_folder = mask_folder

        if train_val == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize(256, 256),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(220),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(256, 256),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.images[idx])
        mask_path = os.path.join(self.mask_folder, self.images[idx])
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert('L')   # todo: use personal dataset -> mask = Image.open(mask_path)

        image, mask = self.transforms(image, mask)

        return image, mask

    def __len__(self):
        return len(self.images)
