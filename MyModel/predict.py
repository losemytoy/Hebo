import os
import time

import cv2
import torch

from MyModel.Models.ResUNet import ResUNet
from MyModel.until import transforms
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from MyModel.Models.AResUNet import AResUNet
from MyModel.Models.UNet import UNet
from MyModel.Models.DeepLabV3plus import DeepLabV3plus
from MyModel.Models.Resnet_Deeplab import Resnet_Deeplab

import tifffile as tif


def main():
    aux = False  # inference time not need aux_classifier
    classes = 1

    # check files
    weights_path = '../save_weights/AResUNet_channel_5c_0.6236.pth'
    # weights_path = './pre_weights/AResUnet_5c_checkpoint_0.6281.pth'
    # img_path = 'D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\Test\\Images\\29.tif'
    img_path = '../Test/Images/6.tif'
    # img_path = '../Test/RGB/000000000040.tif'
    # img_path = '../Test/NIR_SAR/66.tif'
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"weights {img_path} not found."

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # create model
    # model = UNet(3, 1)
    # model = DeepLabV3plus(num_classes=1, backbone="mobilenet", downsample_factor=16, pretrained=False)
    model = AResUNet(num_classes=1)
    # model = ResUNet(num_classes=1)
    # model = Resnet_Deeplab(num_classes=1)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # preprocess image
    img_transforms = transforms.Compose([
        # transforms.Resize(256, 256),
        transforms.ConvertArray(),
        transforms.ToTensor(),
    ])

    original_img = tif.imread(img_path)
    img, s = img_transforms(original_img, torch.rand(1, 256, 256))
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        output = model(img.to(device)).cpu()
        # output,weight = model(img.to(device))
        # output = output.cpu()
        output = output.permute(0, 2, 3, 1)
        output = output.numpy()
        prediction = np.concatenate(output, axis=1)

        # prediction = output.argmax(1).squeeze(0)
        # prediction = prediction.to("cpu").numpy()
        # mask = Image.fromarray(prediction)
        # mask.save("predict_result.png")

    plt.subplot(121)
    original_img = np.transpose(original_img, (2, 0, 1))
    channels = original_img[:3]
    merged_image = np.stack(channels, axis=-1)
    merged_image = merged_image / merged_image.max()
    plt.imshow(merged_image)
    # plt.imsave('D:/OneDrive - The University of Nottingham/Dissertation/Data/Test/Result/ori29.png', merged_image)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(prediction, cmap='gray')
    # plt.show()
    output[output < 0.5] = 0
    output[output >= 0.5] = 1
    output1 = np.squeeze(output)
    # cv2.imwrite('D:/OneDrive - The University of Nottingham/Dissertation/Data/Test/Result/Aresunet_channel_res29.png', output1*255)
    cv2.imwrite('../result/Aresunet_channel_res6.png', output1*255)

if __name__ == '__main__':
    main()
