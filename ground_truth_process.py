from osgeo import gdal
import numpy as np
import os
import cv2
from PIL import Image

load_path = "D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\Dataset\\labels"
save_path = "D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\ResearchData\\Masks"
fileList = os.listdir(load_path)
# target = Image.open(os.path.join(load_path,'000000000001.png'))
# np.set_printoptions(threshold=np.inf)
# print(np.array(target))
# target = target.convert('P')
# target.show()
for i in range(len(fileList)):
    if fileList[i].endswith('.png'):
        dataset = gdal.Open(os.path.join(load_path, fileList[i]))
        img_width = dataset.RasterXSize
        img_height = dataset.RasterYSize
        image = dataset.ReadAsArray(0, 0, img_width, img_height).astype(np.float32)
        del dataset
        image = image/5 * 255

        cv2.imwrite(os.path.join(save_path, fileList[i]), image)
