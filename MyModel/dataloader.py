import os

from torch.utils.data import DataLoader

from dataset import Segmentation

from sklearn.model_selection import train_test_split

root_path = 'D:/OneDrive - The University of Nottingham/Dissertation/Data/ResearchData/'

image_folder = os.path.join(root_path, 'Images')
mask_folder = os.path.join(root_path, 'Masks')

images = os.listdir(image_folder)

train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
test_images, val_images = train_test_split(test_images, test_size=0.5, random_state=42)

train_set = Segmentation(train_images, image_folder, mask_folder, train_val='train')
test_set = Segmentation(test_images, image_folder, mask_folder, train_val='test')
val_set = Segmentation(val_images, image_folder, mask_folder, train_val='val')

train_loader = DataLoader(train_set, batch_size=32,shuffle=True)
test_loader = DataLoader(test_set, batch_size=16,shuffle=True)
val_loader = DataLoader(val_set, batch_size=32,shuffle=True)
