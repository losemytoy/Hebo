import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, dilation=rate, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=rate * 6, dilation=rate * 6, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=rate * 12, dilation=rate * 12, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=rate * 18, dilation=rate * 18, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # global_feature = torch.mean(x, 2, True)
        # global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5(x)
        global_feature = F.interpolate(global_feature, x.shape[2:], mode='bilinear', align_corners=True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result