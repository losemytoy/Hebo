import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from MyModel.Models.ResUNet import ResUNet
from MyModel.Models.UNet import UNet
from MyModel.Models.DeepLabV3plus import DeepLabV3plus


def main():
    aux = False  # inference time not need aux_classifier
    classes = 1

    # check files
    weights_path = './pre_weights/deeplab_checkpoint_0.1396.pth'
    img_path = 'D:/OneDrive - The University of Nottingham/Dissertation/Data/ResearchData/Images/10.png'
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"weights {img_path} not found."

    # get devices
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # create model
    # model = UNet(3, 1)
    model = DeepLabV3plus(num_classes=1, backbone="mobilenet", downsample_factor=16, pretrained=False)
    # model = ResUNet(num_classes=1)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # preprocess image
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(256), #todo deeplabv3+ don't need this operation, resUNet have to crop the size,
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    original_img = Image.open(img_path)
    img = img_transforms(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time.time()
        output = model(img.to(device)).cpu()
        t_end = time.time()
        print("inference time: {}".format(t_end - t_start))

        output = output.permute(0, 2, 3, 1)
        output = output.numpy()
        prediction = np.concatenate(output, axis=1)

        # prediction = output.argmax(1).squeeze(0)
        # prediction = prediction.to("cpu").numpy()
        # mask = Image.fromarray(prediction)
        # mask.save("predict_result.png")

    plt.subplot(121)
    plt.imshow(np.array(original_img))
    plt.subplot(122)
    plt.imshow(prediction, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
