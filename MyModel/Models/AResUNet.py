from torch import nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from einops import rearrange, repeat


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='pixelshuffle'):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        # define self-attention
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU()
        self.attn = GlobalLocalAttention(dim=mid_channels, num_heads=8, window_size=8)
        if self.upsample_mode == 'conv_transpose':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3,
                                               stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode == 'pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=2)

        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.attn(x)
        x = self.upsample(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class BasicConv(nn.Module):
    def __init__(
            self,
            in_planes,
            out_planes,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            relu=True,
            bn=True,
            bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

    """
        paper: https://arxiv.org/pdf/2010.03045.pdf
        source code: https://github.com/LandskapeAI/triplet-attention
    """


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.fc(self.avg_pool(x))
        # max_out = self.fc(self.max_pool(x))
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = avg_out + max_out
        cc = self.sigmoid(out)
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True,
                 use_triplet_attention=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        if use_triplet_attention:
            self.triplet_attention = TripletAttention()
        else:
            self.triplet_attention = None
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))
        self.reduce = nn.Sequential(
            # nn.Conv2d(dim, dim, kernel_size=1),
            # nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

            self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def calMask(self, x, window_size, shift_size):
        _, _, H, W = x.size()
        img_mask = torch.zeros((1, 1, H, W))  # 1 1 H W
        h_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)  # nW, 1, window_size, window_size
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def com(self, x, B, C, Hp, Wp):

        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale  # (4,8,64,64)

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)
        return dots, v

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)
        # global => q, k, v
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        dots, v = self.com(x, B, C, Hp, Wp)

        if not self.triplet_attention is None:
            dots = self.triplet_attention(dots)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        # shifted_x = torch.roll(attn, shifts=(-self.ws, -self.ws), dims=(2, 3))
        # shifted_x = self.pad(shifted_x, self.ws)
        # B, C, Hp, Wp = shifted_x.shape
        # dots, v = self.com(shifted_x, B, C, Hp, Wp)
        # mask = self.calMask(attn, self.ws, self.ws // 2)
        # nW = mask.shape[0]
        # mask = mask.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # dots = dots.view(B, nW, self.num_heads, self.ws*self.ws, self.ws*self.ws) + mask.unsqueeze(1).unsqueeze(0)
        # dots = dots.view(-1, self.num_heads, self.ws*self.ws, self.ws*self.ws)
        # attn = dots.softmax(dim=-1)
        # attn = attn @ v
        # attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
        #                  d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)
        #
        # out = attn[:, :, :H, :W]

        avg_pool_w = F.avg_pool2d(attn, (self.ws//2, 1), stride=(self.ws//2, 1))
        avg_pool_h = F.avg_pool2d(attn, (1, self.ws//2), stride=(1, self.ws//2))
        max_pool_w = F.max_pool2d(attn, (self.ws//2, 1), stride=(self.ws//2, 1))
        max_pool_h = F.max_pool2d(attn, (1, self.ws//2), stride=(1, self.ws//2))
        Fw = avg_pool_w + max_pool_w  # (1,256,1,64)
        Fh = avg_pool_h + max_pool_h  # (1,256,64,1)
        Fw = self.reduce(Fw)
        Fh = self.reduce(Fh)
        x = Fh @ Fw
        out = attn * x

        # out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
        #       self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class Refine(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(in_channels, in_channels // 16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(in_channels // 16, in_channels, kernel_size=1),
                                nn.Sigmoid())
        self.sc = SeparableConvBN(in_channels, in_channels, kernel_size=3)
        self.fin = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.sa(x) + self.ca(x)
        out = self.sc(out)
        return self.fin(out)


class AResUNet(nn.Module):
    def __init__(self, num_classes):
        super(AResUNet, self).__init__()
        resnet = models.resnet50()
        filters = [64, 256, 512, 1024, 2048]
        # self.ca = ChannelAttention(5)
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu1 = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.attn1 = GlobalLocalAttention(dim=256, num_heads=8, window_size=8, use_triplet_attention=False)
        self.attn2 = GlobalLocalAttention(dim=512, num_heads=8, window_size=8, use_triplet_attention=False)
        self.attn3 = GlobalLocalAttention(dim=1024, num_heads=8, window_size=8, use_triplet_attention=False)

        self.bridge = DecoderBlock(in_channels=filters[3], mid_channels=filters[3] * 4, out_channels=filters[3])
        self.decoder1 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2])
        self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
                                     out_channels=filters[1])
        self.decoder3 = DecoderBlock(in_channels=filters[1] + filters[0], mid_channels=filters[0] * 4,
                                     out_channels=filters[0])
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Refine(32),
            nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # weight = self.ca(x)
        # x *= weight
        x = self.conv1(x)  # (1,64,128,128)
        x = self.bn1(x)     # (1,64,128,128)
        x = self.relu1(x)   # (1,64,128,128)
        x_ = self.maxpool(x)    # (1,64,64,64)

        # (1,64,64,64)
        e1 = self.encoder1(x_)  # (1,256,64,64)
        e1 = self.attn1(e1)  # (1,256,64,64)
        e2 = self.encoder2(e1)  # (1,512,32,32)
        e2 = self.attn2(e2)     # (1,512,32,32)
        e3 = self.encoder3(e2)  # (1,1024,16,16)
        e3 = self.attn3(e3)     # (1,1024,16,16)
        bridge = self.bridge(e3)      # (1,1024,32,32)

        d2 = self.decoder1(torch.cat([bridge, e2], dim=1))  # (1,512,64,64)
        d3 = self.decoder2(torch.cat([d2, e1], dim=1))  # (1,256,128,128)
        d4 = self.decoder3(torch.cat([d3, x], dim=1))  # (1,64,256,256)
        return self.final(d4)   #(1,1,256,256)


if __name__ == '__main__':
    model = AResUNet(num_classes=1)
    model.eval()
    input = torch.rand(1, 5, 256, 256)
    output = model(input)
    print(output.size())
    # print(weight)
