import torch.nn as nn
import torch.nn.functional as F
import torch

from ssd.layers import L2Norm
from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url

model_urls = {
    'vgg': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
}

# output dimension = (input dimension - kernel_size + 2*padding) / stride + 1

# borrowed from https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py

#300,300,3 -> 300,300,64 -> 300,300,64 -> 150,150,64-> 150,150,128 -> 150,150,128 -> 75,75,128 -> 75,75,256 -> 75,75,256 -> 75,75,256 -> 
#38,38,256 -> 38,38,512 -> 38,38,512 -> 38,38,512 -> 19,19,512 -> 19,19,512 -> 19,19,512 -> 19,19,512

def add_vgg(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)# 19,19,512->19,19,512
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)#19,19,512->19,19,1024
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)#19,19,1024->19,19,1024
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

#19,19,1024 -> 19,19,256 -> 10,10,512 -> 10,10,512 -> 10,10,128 -> 5,5,256 -> 5,5,256 -> 5,5,128 -> 3,3,256 -> 3,3,128 -> 1,1,256

def add_extras(cfg, i, size=300):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers


vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras_base = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}


class FSSD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size = cfg['INPUT']['IMAGE_SIZE']
        vgg_config = vgg_base[str(size)]
        extras_config = extras_base[str(size)]
        self.vgg = nn.ModuleList(add_vgg(vgg_config))
        self.extras = nn.ModuleList(add_extras(extras_config, i=1024, size=size))
        self.conv4_3_extra = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7_extra = nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8_2_extra = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=4, padding=1, output_padding=1)
        self.l2_norm_conv4_3 = L2Norm(512, scale=20)
        self.l2_norm = L2Norm(256, scale=20)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def init_from_pretrain(self, state_dict):
        self.vgg.load_state_dict(state_dict)

    def forward(self, x):
        features = []
        for i in range(23):
            x = self.vgg[i](x)
        conv4_3 = self.conv4_3_extra(x)
        conv4_3 = self.l2_norm_conv4_3(conv4_3)
        conv4_3 = F.relu(conv4_3,inplace=True)

        # apply vgg up to fc7
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        conv7 = self.conv7_extra(x)
        conv7 = self.l2_norm(conv7)
        conv7 = F.relu(conv7,inplace=True)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k== 1:
                conv8_2 = self.conv8_2_extra(x)
                conv8_2 = self.l2_norm(conv8_2)
                conv8_2 = F.relu(conv8_2,inplace=True)
                fused_feature = torch.cat((conv4_3,conv7,conv8_2),1)
                features.append(fused_feature)
            if k % 2 == 1:
                features.append(x)

        return tuple(features)
        
@registry.BACKBONES.register('FSSD')
def fssd(cfg, pretrained=True):
    model = FSSD(cfg)
    if pretrained:
        model.init_from_pretrain(load_state_dict_from_url(model_urls['vgg']))
    return model
