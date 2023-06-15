from osgeo import gdal
import numpy as np
import os
import cv2

load_path = "D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\Dataset\\images"
save_path = "D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\ResearchData\\images"
fileList = os.listdir(load_path)

for i in range(len(fileList)):
    if fileList[i].endswith('.png'):
        dataset = gdal.Open(os.path.join(load_path, fileList[i]))
        img_width = dataset.RasterXSize
        img_height = dataset.RasterYSize
        image = dataset.ReadAsArray(0, 0, img_width, img_height).astype(np.float)
        del dataset
        image = image.swapaxes(0, 2)
        image = image.swapaxes(0, 1)

        cv2.imwrite(os.path.join(save_path, fileList[i]), image)
