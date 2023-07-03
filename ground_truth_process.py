from osgeo import gdal
import numpy as np
import os
import cv2
from PIL import Image
import shutil

# load_path = "D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\Dataset\\labels"
# load_path = "D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\Water_Bodies_Dataset\\Masks"
# save_path = "D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\ResearchData\\Masks"
# fileList = os.listdir(load_path)
# target = Image.open(os.path.join(load_path,'000000000001.png'))
# np.set_printoptions(threshold=np.inf)
# print(np.array(target))
# target = target.convert('P')
# target.show()

# existing_files = os.listdir(save_path)
# existing_count = len(existing_files)
# i = 0
# for file_name in fileList:
#     if file_name.endswith('.png') or file_name.endswith('.jpg'):
#         image = Image.open(os.path.join(load_path, file_name))
#
#         new_index = existing_count + i
#         new_file_name = '{}.{}'.format(new_index, file_name.split('.')[-1])
#
#         image.save(os.path.join(save_path, new_file_name))
#         i += 1

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

path = 'D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\5_channels_dataset' #待读取文件的文件夹绝对地址
save_path = "D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\5_channels_dataset\\Masks"
existing_files = os.listdir(save_path)
existing_count = len(existing_files)
i = 0
files = os.listdir(path) #获得文件夹中所有文件的名称列表
list0 = [] #存放path路径中的文件内容
list1 = [] #存放path中子文件夹的文件内容
for file in files:
    path1 = path+"\\"+file
    files1 = os.listdir(path1)
    for file1 in files1:
        if file1 == "labels":
            path2 = path1 + "\\" + file1
            files2 = os.listdir(path2)
            for file2 in files2:
                if file2.endswith('.tif'):
                    dataset = gdal.Open(os.path.join(path2, file2))
                    img_width = dataset.RasterXSize
                    img_height = dataset.RasterYSize
                    image = dataset.ReadAsArray(0, 0, img_width, img_height).astype(np.float32)
                    del dataset
                    image = image * 255

                    new_index = existing_count + i
                    new_file_name = '{}.{}'.format(new_index, file2.split('.')[-1])

                    cv2.imwrite(os.path.join(save_path, new_file_name), image)
                    i += 1
                    #
                    # new_index = existing_count + i
                    # new_file_name = '{}.{}'.format(new_index, file2.split('.')[-1])
                    # shutil.copy(os.path.join(path2, file2), os.path.join(save_path, new_file_name))
                    # i += 1