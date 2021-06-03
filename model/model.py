import numpy as np

import torch
import torch.nn as nn
import pretrainedmodels
from efficientnet_pytorch import EfficientNet

available_models = [
    "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4", 
    "efficientnet-b5", "efficientnet-b6", "efficientnet-b7", 
    "alexnet", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19_bn", "vgg19",
    "densenet121", "densenet169", "densenet201", "densenet161",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext101_32x4d", "resnext101_64x4d", 
    "squeezenet1_0", "squeezenet1_1", "nasnetamobile", "nasnetalarge", 
    "dpn68", "dpn68b", "dpn92", "dpn98", "dpn131", 
    "senet154", "se_resnet50", "se_resnet101", "se_resnet152", "se_resnext50_32x4d", "se_resnext101_32x4d",
    "inceptionv4", "inceptionresnetv2", "xception", "fbresnet152", "bninception",
    "cafferesnet101", "pnasnet5large", "polynet"
]

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
        
        if "efficientnet" in base_model_name:
            self.base_model = EfficientNet.from_pretrained(base_model_name)
            self.feature_extractor = self.base_model.extract_features
            
        else:
            self.base_model = pretrainedmodels.__dict__[base_model_name](num_classes=1000)
            if "_features" in dir(self.base_model):
                self.feature_extractor = self.base_model._features
            else:
                self.feature_extractor = self.base_model.features
        
        if freeze: 
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.base_model.eval()
        with torch.no_grad():
            test_input  = torch.empty(1, 3, 224, 224)
            dims = self.feature_extractor(test_input).size()
        self.base_model.train()

        layer_list = []
        if len(dims)>2:
            layer_list.append(nn.AdaptiveAvgPool2d((1, 1)))
            layer_list.append(nn.Flatten())
        layer_list.append(nn.Dropout())
        layer_list.append(nn.Linear(dims[1], num_classes))
        
        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
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