import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from MyModel.Models.ASPP import ASPP

from MyModel.Models.BackBone.xception_deeplab import xception


class DeepLabV3plus(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=False, downsample_factor=16):
        super(DeepLabV3plus, self).__init__()
        if backbone == "mobilenet":
            self.backbone = models.mobilenet_v2(weights="IMAGENET1K_V1").features
            in_channel = 320
            low_level_channels = 24
        # elif backbone == "xception":
        #     self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
        #     in_channels = 2048
        #     low_level_channels = 256
        else:
            raise ValueError('model need a valid backbone')

        self.aspp = ASPP(dim_in=in_channel, dim_out=256, rate= 16//downsample_factor)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 3)

    def get_features(self, x):
        feature = []
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            feature.append(x)
        return feature

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        features = self.get_features(x)
        low_level_features = features[2]
        x = features[-2]
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = torch.sigmoid(x)
        return x
