import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models
from utils import save_net,load_net
from layer import convDU,convLR
from masksembles.torch import Masksembles2D
import numpy as np 


class CANNet2s(nn.Module):
    def __init__(self, load_weights=False):
        super(CANNet2s, self).__init__()
        self.context1 = ContextualModule(64, 64)
        self.context2 = ContextualModule(128, 128)
        self.context3 = ContextualModule(256, 256)
        self.context4 = ContextualModule(512, 512)
        # self.range = 0.0217
        self.encoder = ResNet18(ResidualBlock)
        self.decoder = Decoder()
        self.output_layer = nn.Conv2d(64, 10, kernel_size=1)
        self.relu = nn.ReLU()


    def forward(self,x_prev,x):
        # print(x_prev.size())
        x_prev = self.encoder(x_prev)
        # print(x_prev.size())
        x_prev = self.context4(x_prev)

        x = self.encoder(x)
        x = self.context4(x)

        x = torch.cat((x_prev,x),1)
        
        # print(x.size())
        x = self.decoder(x)
        # print(x.size())
        x = self.output_layer(x)
        x = self.relu(x)
        # x = x.double()
        # x = torch.where(x > self.range, x, 0.0)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ContextualModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(ContextualModule, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features,features,kernel_size=1)

    def __make_weight(self,feature,scale_feature):
        weight_feature = feature - scale_feature
        return F.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        multi_scales = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.scales]
        weights = [self.__make_weight(feats,scale_feature) for scale_feature in multi_scales]
        overall_features = [(multi_scales[0]*weights[0]+multi_scales[1]*weights[1]+multi_scales[2]*weights[2]+multi_scales[3]*weights[3])/(weights[0]+weights[1]+weights[2]+weights[3])]+ [feats]
        bottle = self.bottleneck(torch.cat(overall_features, 1))
        return self.relu(bottle)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        # self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out


class Decoder(nn.Module):

    def __init__(self, out_channels = 64):
        super(Decoder, self).__init__()

        features = out_channels

        self.conv1 = nn.Conv2d(features * 16, features * 8, kernel_size=3, padding=2, dilation = 2)
        self.conv2 = nn.Conv2d(features * 8, features * 8, kernel_size=3, padding=2, dilation = 2)
        self.conv3 = nn.Conv2d(features * 8, features * 8, kernel_size=3, padding=2, dilation = 2)
        self.conv4 = nn.Conv2d(features * 8, features * 4, kernel_size=3, padding=2, dilation = 2)
        self.conv5 = nn.Conv2d(features * 4, features * 2, kernel_size=3, padding=2, dilation = 2)
        self.conv6 = nn.Conv2d(features * 2, out_channels, kernel_size=3, padding=2, dilation = 2)

        # self.pool1 = nn.AvgPool2d(kernel_size = 2, stride = 2)

        self.batchNorm1 = nn.BatchNorm2d(features * 8)
        self.batchNorm2 = nn.BatchNorm2d(features * 8)
        self.batchNorm3 = nn.BatchNorm2d(features * 8)
        self.batchNorm4 = nn.BatchNorm2d(features * 4)
        self.batchNorm5 = nn.BatchNorm2d(features * 2)
        self.batchNorm6 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU(inplace = True)

    def forward(self, x):
        dec1 = self.conv1(x)
        dec2 = self.batchNorm1(dec1)
        dec3 = self.act(dec2)

        dec4 = self.conv2(dec3)
        dec5 = self.batchNorm2(dec4)
        dec6 = self.act(dec5)
        dec7 = self.conv3(dec6)
        dec8 = self.batchNorm3(dec7)
        dec9 = self.act(dec8)
        dec10 = self.conv4(dec9)
        dec11 = self.batchNorm4(dec10)
        dec12 = self.act(dec11)

        dec13 = self.conv5(dec12)
        dec14 = self.batchNorm5(dec13)
        dec15 = self.act(dec14)

        dec16 = self.conv6(dec15)
        dec17 = self.batchNorm6(dec16)
        dec18 = self.act(dec17)
        # print(dec18.size())

        return dec18


# model = CANNet2s()
# # model = model.cuda()
# x_prev = torch.randn(2,3,360,640)
# x = torch.randn(2,3,360,640)
# model(x_prev,x)