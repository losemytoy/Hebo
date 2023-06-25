from osgeo import gdal
import numpy as np
import os
from PIL import Image

load_path = "D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\Water_Bodies_Dataset\\Images"
# load_path = "D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\Dataset\\images"
save_path = "D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\ResearchData\\Images"
fileList = os.listdir(load_path)

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
# for i in range(len(fileList)):
#     if fileList[i].endswith('.png') or fileList[i].endswith('.jpg'):
#         dataset = gdal.Open(os.path.join(load_path, fileList[i]))
#         img_width = dataset.RasterXSize
#         img_height = dataset.RasterYSize
#         image = dataset.ReadAsArray(0, 0, img_width, img_height).astype(np.float)
#         del dataset
#         image = image.swapaxes(0, 2)
#         image = image.swapaxes(0, 1)

        # cv2.imwrite(os.path.join(save_path, fileList[i]), image)
