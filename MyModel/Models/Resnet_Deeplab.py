import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from MyModel.Models.ASPP import ASPP
from MyModel.Models.BackBone.resnet import resnet50


class Resnet_Deeplab(nn.Module):
    def __init__(self, num_classes, pretrained=False, downsample_factor=16):
        super(Resnet_Deeplab, self).__init__()
        self.backbone = resnet50(aux=False)
        self.backbone2 = resnet50(aux=True)
        in_channel = 2048
        low_level_channels = 320

        self.aspp = ASPP(dim_in=in_channel, dim_out=256, rate=16 // downsample_factor)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(64 + 512, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 3)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, y = x.split([4, 1], dim=1)
        x, low_level_feature1, low_level_feature2 = self.backbone(x)  # (1,2048,8,8) (1,64,64,64) (1,256,64,64)
        y, y_low_level_feature1, y_low_level_feature2 = self.backbone2(y)  # (1,2048,8,8) (1,64,64,64) (1,256,64,64)
        x = self.aspp(x) #(1,256,8,8)
        y = self.aspp(y) #(1,256,8,8)
        low_level_features = self.shortcut_conv(torch.cat((low_level_feature1, low_level_feature2), dim=1))
        # todo  multiple channel fusion module to combine 'optical' and 'SAR' low level features
        x = torch.cat((x, y), dim=1)
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    model = Resnet_Deeplab(num_classes=1)
    model.eval()
    input = torch.rand(1, 5, 256, 256)
    output = model(input)
    print(output.size())