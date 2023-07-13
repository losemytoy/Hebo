import torch.nn as nn
from collections import OrderedDict

class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)

            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

# class MultipleChannelsFusion(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=2):
#         super(MultipleChannelsFusion, self).__init__()
#         self.conv1 = nn.Conv2d()
#         self.gate_channels = gate_channels
#         self.mlp = nn.Sequential(
#             Flatten(),
#             nn.Linear(gate_channels, gate_channels // reduction_ratio),
#             nn.ReLU()
#         )
#         self.pool_types = pool_types
#         self.incr = nn.Linear(gate_channels // reduction_ratio, gate_channels)
#
#     def forward(self, x):
#         avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#         max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#         avgpoolmlp = self.mlp(avg_pool)
#         maxpoolmlp=self.mlp(max_pool)
#         pooladd = avgpoolmlp+maxpoolmlp
#
#         self.pool = SoftPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#         soft_pool = self.mlp(self.pool(x))
#         weightPool = soft_pool * pooladd
#         # channel_att_sum = self.mlp(weightPool)
#         channel_att_sum = self.incr(weightPool)
#         Att = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
#         return Att
