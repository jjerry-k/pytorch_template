import numpy as np

import torch
import torch.nn as nn

import timm

available_models = timm.list_models()

class Model(nn.Module):
    """
    Thi is example code.
    Customize your self.
    """
    def __init__(self, base_model_name="alexnet", num_classes=10, freeze=True):
        super(Model, self).__init__()

        assert base_model_name in available_models, f"Available pretrained model list: {available_models}"
        
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        self.freeze = freeze
        
        self.model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=10)
        
        if freeze: 
            for layer_name, ops in self.model.named_children():
                if layer_name != 'classifier':
                    ops.requires_grad = False
        
    def forward(self, x):
        x = self.model(x)
        return x

# class Conv_Block(nn.Module):
#     '''(Conv, ReLU) * 2'''
#     def __init__(self, in_ch, out_ch, pool=None):
#         super(Conv_Block, self).__init__()
#         layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1),
#                   nn.ReLU(inplace=True),
#                   nn.Conv2d(out_ch, out_ch, 3, padding=1),
#                   nn.ReLU(inplace=True)]
        
#         if pool:
#             layers.insert(0, nn.MaxPool2d(2, 2))
        
#         self.conv = nn.Sequential(*layers)
            

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class Upconv_Block(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(Upconv_Block, self).__init__()

#         self.upconv = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        
#         self.conv = Conv_Block(in_ch, out_ch)

#     def forward(self, x1, x2):
#         # x1 : unpooled feature
#         # x2 : encoder feature
#         x1 = self.upconv(x1)
#         x1 = nn.UpsamplingBilinear2d(x2.size()[2:])(x1)
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x

# class Model(nn.Module):
#     def __init__(self, input_channel=3, num_classes=5):
#         super(Model, self).__init__()
#         self.conv1 = Conv_Block(input_channel, 64)
#         self.conv2 = Conv_Block(64, 128, pool=True)
#         self.conv3 = Conv_Block(128, 256, pool=True)
#         self.conv4 = Conv_Block(256, 512, pool=True)
#         self.conv5 = Conv_Block(512, 1024, pool=True)
        
#         self.unconv4 = Upconv_Block(1024, 512)
#         self.unconv3 = Upconv_Block(512, 256)
#         self.unconv2 = Upconv_Block(256, 128)
#         self.unconv1 = Upconv_Block(128, 64)
        
#         self.prediction = nn.Conv2d(64, num_classes, 1)
        
#     def forward(self, x):
#         en1 = self.conv1(x) #/2
#         en2 = self.conv2(en1) #/4
#         en3 = self.conv3(en2) #/8
#         en4 = self.conv4(en3) #/16
#         en5 = self.conv5(en4) 
        
#         de4 = self.unconv4(en5, en4) # /8
#         de3 = self.unconv3(de4, en3) # /4
#         de2 = self.unconv2(de3, en2) # /2
#         de1 = self.unconv1(de2, en1) # /1
        
#         output = self.prediction(de1)
#         return output