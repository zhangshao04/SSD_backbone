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

class AttentionModule_first_stage_conv4_3(nn.Module):
    def __init__(self, channels):
        super(AttentionModule_first_stage_conv4_3, self).__init__()
        self.trunk_branch = TrunkBranch(channels)
        self.mask_branch_19 = MaskBranch19(channels)
        self.mask_branch_19_30 = MaskBranch19_30(channels)
        self.mask_branch_30_58 = MaskBranch30_58_conv4_3(channels)
        self.mask_branch_58_68 = MaskBranch58_68_conv4_3(channels)
        self.mask_branch_68_78 = MaskBranch68_78_conv4_3(channels)
        self.mask_branch_out = MaskBranchout(channels)
        self.residual_blocks = ResidualBlock(channels)
        self.l2norm = L2Norm(channels,scale=20)

    def forward(self, x):

        x = self.residual_blocks(x)
        # Trunk branch path
        out_trunk = self.trunk_branch(x)

        x_mask_19_out = self.mask_branch_19(x)
        x_mask_30_out = self.mask_branch_19_30(x_mask_19_out)
        x_mask_58_out = self.mask_branch_30_58(x_mask_30_out)
        x1 = x_mask_30_out + x_mask_58_out
        x_mask_68_out = self.mask_branch_58_68(x1)
        x2 = x_mask_68_out + x_mask_19_out
        x_mask_78_out = self.mask_branch_68_78(x2)
        out_mask = self.mask_branch_out(x_mask_78_out) 

        # Applying the attention mask
        out = out_trunk * (1 + out_mask)
        
        # Residual blocks
        out = self.residual_blocks(out)
        out = self.l2norm(out)
        out = F.relu(out,inplace=True)

        return out
    

class AttentionModule_first_stage_conv7(nn.Module):
    def __init__(self, channels):
        super(AttentionModule_first_stage_conv7, self).__init__()
        self.trunk_branch = TrunkBranch(channels)
        self.mask_branch_19 = MaskBranch19(channels)
        self.mask_branch_19_30 = MaskBranch19_30(channels)
        self.mask_branch_30_58 = MaskBranch30_58_conv7(channels)
        self.mask_branch_58_68 = MaskBranch58_68_conv7(channels)
        self.mask_branch_68_78_conv4_3 = MaskBranch68_78_conv7(channels)
        self.mask_branch_out = MaskBranchout(channels)
        self.residual_blocks = ResidualBlock(channels)
        self.residual_blocks = ResidualBlock(channels)
        self.l2norm = L2Norm(channels,scale=20)

    def forward(self, x):

        x = self.residual_blocks(x)
        # Trunk branch path
        out_trunk = self.trunk_branch(x)

        x_mask_19_out = self.mask_branch_19(x)
        x_mask_30_out = self.mask_branch_19_30(x_mask_19_out)
        x_mask_58_out = self.mask_branch_30_58(x_mask_30_out)
        x1 = x_mask_30_out + x_mask_58_out
        x_mask_68_out = self.mask_branch_58_68(x1)
        x2 = x_mask_68_out + x_mask_19_out
        x_mask_78_out = self.mask_branch_68_78_conv4_3(x2)
        out_mask = self.mask_branch_out(x_mask_78_out) 

        # Applying the attention mask
        out = out_trunk * (1 + out_mask)
        
        # Residual blocks
        out = self.residual_blocks(out)
        out = self.l2norm(out)
        out = F.relu(out,inplace=True)

        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1)

    def forward(self, x):
        y = x
        x = self.batch_norm(x)
        x = F.relu(x, inplace=True)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x, inplace=True)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x, inplace=True)
        x = self.conv(x)
        x += y

        return x
    
class TrunkBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.residual_block = ResidualBlock(channels)

    def forward(self,x):
        x = self.residual_block(x)
        x = self.residual_block(x)

        return x

class MaskBranch19(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.residual_block = ResidualBlock(channels)

    def forward(self,x):
        x = self.maxpool(x)
        x = self.residual_block(x)
        x = self.residual_block(x)
        
        return x

class MaskBranch19_30(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.residual_block = ResidualBlock(channels)

    def forward(self,x):
        x = self.maxpool(x)
        x = self.residual_block(x)
        x = self.residual_block(x)

        return x

class MaskBranch30_58_conv4_3(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.residual_block = ResidualBlock(channels)

    def forward(self,x):
        x = self.maxpool(x)
        x = self.residual_block(x)
        x = self.residual_block(x)
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)

        return x
    
class MaskBranch30_58_conv7(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.residual_block = ResidualBlock(channels)

    def forward(self,x):
        x = self.maxpool(x)
        x = self.residual_block(x)
        x = self.residual_block(x)
        x = F.interpolate(x,size=(5,5),mode='bilinear',align_corners=False)

        return x

class MaskBranch58_68_conv7(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.residual_block = ResidualBlock(channels)

    def forward(self,x):
        x = self.residual_block(x)
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)

        return x
    
class MaskBranch58_68_conv4_3(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.residual_block = ResidualBlock(channels)

    def forward(self,x):
        x = self.residual_block(x)
        x = F.interpolate(x,size=(19,19),mode='bilinear',align_corners=False)

        return x

class MaskBranch68_78_conv4_3(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.residual_block = ResidualBlock(channels)

    def forward(self,x):
        x = self.residual_block(x)
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)

        return x
    
class MaskBranch68_78_conv7(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.residual_block = ResidualBlock(channels)

    def forward(self,x):
        x = self.residual_block(x)
        x = F.interpolate(x,size=(19,19),mode='bilinear',align_corners=False)

        return x

class MaskBranchout(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self,x):
        x = self.batch_norm(x)
        x = F.relu(x,inplace=True)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x,inplace=True)
        x = self.conv(x)
        x = self.sigmod(x)

        return x


class FASSD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size = cfg['INPUT']['IMAGE_SIZE']
        vgg_config = vgg_base[str(size)]
        extras_config = extras_base[str(size)]
        self.vgg = nn.ModuleList(add_vgg(vgg_config))
        self.extras = nn.ModuleList(add_extras(extras_config, i=1024, size=size))
        self.conv4_3_extra = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7_extra = nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8_2_extra_conv4_3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=4, padding=1, output_padding=1)
        self.conv7_extra_target = nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=1)
        self.conv8_2_extra_conv7 = nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1)
        self.conv9_2_extra = nn.ConvTranspose2d(256,512,kernel_size=3,stride=4)
        self.l2_norm_512 = L2Norm(512, scale=20)
        self.l2_norm_1024 = L2Norm(1024, scale=20)
        self.l2_norm_256 = L2Norm(256,scale=20)
        self.attention_module_conv4_3 = AttentionModule_first_stage_conv4_3(512)
        self.attention_module_conv7 = AttentionModule_first_stage_conv7(1024)

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
        conv4_3_attention_module = self.attention_module_conv4_3(x)
        conv4_3_fused_target = self.conv4_3_extra(conv4_3_attention_module)
        conv4_3_fused_target = self.l2_norm_512(conv4_3_fused_target)
        conv4_3_fused_target = F.relu(conv4_3_fused_target,inplace=True) #计算出conv4_3的目标特征图
        # apply vgg up to fc7
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        conv7_attention_module = self.attention_module_conv7(x)
        conv7_fused_target = self.conv7_extra_target(conv7_attention_module)
        conv7_fused_target = self.l2_norm_1024(conv7_fused_target)
        conv7_fused_target = F.relu(conv7_fused_target,inplace=True)#计算出conv7的目标特征图
        conv7_fused = self.conv7_extra(x)
        conv7_fused = self.l2_norm_256(conv7_fused)
        conv7_fused = F.relu(conv7_fused,inplace=True)#计算出conv7融合进conv4_3特征图


        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k == 1:
                conv8_2_fused_conv4_3 = self.conv8_2_extra_conv4_3(x)
                conv8_2_fused_conv4_3 = self.l2_norm_256(conv8_2_fused_conv4_3)
                conv8_2_fused_conv4_3 = F.relu(conv8_2_fused_conv4_3,inplace=True)#计算conv8_2融合进conv4_3特征图
                fused_conv4_3_target = torch.cat((conv4_3_fused_target,conv7_fused,conv8_2_fused_conv4_3),1)#conv4_3融合
                features.append(fused_conv4_3_target)
                conv8_2_fused_conv7 = self.conv8_2_extra_conv7(x)
                conv8_2_fused_conv7 = self.l2_norm_512(conv8_2_fused_conv7)
                conv8_2_fused_conv7 = F.relu(conv8_2_fused_conv7,inplace=True)#计算conv8_2融合进conv7特征图
            if k == 3:
                conv9_2 = self.conv9_2_extra(x)
                conv9_2 = self.l2_norm_512(conv9_2)
                conv9_2 = F.relu(conv9_2,inplace=True)#计算conv9_2融合进conv7特征图
                fused_conv7_target = torch.cat((conv7_fused_target,conv8_2_fused_conv7,conv9_2),1)#conv7融合
                features.append(fused_conv7_target)
            if k % 2 == 1:
                features.append(x)

        return tuple(features)
        
@registry.BACKBONES.register('FASSD')
def fssd(cfg, pretrained=True):
    model = FASSD(cfg)
    if pretrained:
        model.init_from_pretrain(load_state_dict_from_url(model_urls['vgg']))
    return model
