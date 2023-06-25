from osgeo import gdal
import numpy as np
import os
import cv2
from PIL import Image

# load_path = "D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\Dataset\\labels"
load_path = "D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\Water_Bodies_Dataset\\Masks"
save_path = "D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\ResearchData\\Masks"
fileList = os.listdir(load_path)
# target = Image.open(os.path.join(load_path,'000000000001.png'))
# np.set_printoptions(threshold=np.inf)
# print(np.array(target))
# target = target.convert('P')
# target.show()

existing_files = os.listdir(save_path)
existing_count = len(existing_files)
i = 0
for file_name in fileList:
    if file_name.endswith('.png') or file_name.endswith('.jpg'):
        image = Image.open(os.path.join(load_path, file_name))

        new_index = existing_count + i
        new_file_name = '{}.{}'.format(new_index, file_name.split('.')[-1])

        image.save(os.path.join(save_path, new_file_name))
        i += 1

# for file_name in fileList:
#     if file_name.endswith('.png') or file_name.endswith('.jpg'):
#         dataset = gdal.Open(os.path.join(load_path, file_name))
#         img_width = dataset.RasterXSize
#         img_height = dataset.RasterYSize
#         image = dataset.ReadAsArray(0, 0, img_width, img_height).astype(np.float32)
#         del dataset
#         image = image * 255
#
#         new_index = existing_count + i
#         new_file_name = '{}.{}'.format(new_index, file_name.split('.')[-1])
#
#         cv2.imwrite(os.path.join(save_path, new_file_name), image)
#         i += 1