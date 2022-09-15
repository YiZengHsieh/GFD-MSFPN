import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models
from utils import save_net,load_net
from layer import convDU,convLR
from masksembles.torch import Masksembles2D


class CANNet2s(nn.Module):
    def __init__(self, load_weights=False):
        super(CANNet2s, self).__init__()
        self.context1 = ContextualModule(64, 64)
        self.context2 = ContextualModule(128, 128)
        self.context3 = ContextualModule(256, 256)
        self.context4 = ContextualModule(512, 512)

        self.encoder = Encoder(3,64)
        self.decoder = Decoder()

        self.output_layer = nn.Conv2d(64, 10, kernel_size=1)
        self.relu = nn.ReLU()


    def forward(self,x_prev,x):

        x_prev4, x_prev9, x_prev16, x_prev23 = self.encoder(x_prev)
        # x_prev4 = self.context1(x_prev4)
        # x_prev9 = self.context2(x_prev9)
        # x_prev16 = self.context3(x_prev16)
        x_prev = self.context4(x_prev23)

        x4, x9, x16, x23 = self.encoder(x)
        # x4 = self.context1(x4)
        # x9 = self.context2(x9)
        # x16 = self.context3(x16)
        x = self.context4(x23)

        # x4 = torch.cat((x_prev4,x4),1)
        # x9 = torch.cat((x_prev9,x9),1)
        # x16 = torch.cat((x_prev16,x16),1)
        x = torch.cat((x_prev,x),1)
        
        # print(x.size())
        x = self.decoder(x)
        # print(x.size())
        x = self.output_layer(x)
        x = self.relu(x)
        # print(x.size())

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


class Encoder(nn.Module):
    
    def __init__(self, in_channels = 3, init_features = 64, classify = False):
        super(Encoder, self).__init__()

        self.classify = classify

        features = init_features

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = features, kernel_size = 3, padding = 1, dilation = 1)
        self.conv2 = nn.Conv2d(in_channels = features, out_channels = features, kernel_size = 3, padding = 1, dilation = 1)
        self.conv3 = nn.Conv2d(in_channels = features, out_channels = features * 2, kernel_size = 3, padding = 1, dilation = 1)
        self.conv4 = nn.Conv2d(in_channels = features * 2, out_channels = features * 2, kernel_size = 3, padding = 1, dilation = 1)
        self.conv5 = nn.Conv2d(in_channels = features * 2, out_channels = features * 4, kernel_size = 3, padding = 1, dilation = 1)
        self.conv6 = nn.Conv2d(in_channels = features * 4, out_channels = features * 4, kernel_size = 3, padding = 1, dilation = 1)
        self.conv7 = nn.Conv2d(in_channels = features * 4, out_channels = features * 4, kernel_size = 3, padding = 1, dilation = 1)
        self.conv8 = nn.Conv2d(in_channels = features * 4, out_channels = features * 8, kernel_size = 3, padding = 1, dilation = 1)
        self.conv9 = nn.Conv2d(in_channels = features * 8, out_channels = features * 8, kernel_size = 3, padding = 1, dilation = 1)
        self.conv10 = nn.Conv2d(in_channels = features * 8, out_channels = features * 8, kernel_size = 3, padding = 1, dilation = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.act = nn.ReLU(inplace = True)

    def forward(self, x):
            enc1 = self.conv1(x)
            enc2 = self.act(enc1)
            enc3 = self.conv2(enc2)
            enc4 = self.act(enc3)

            enc5 = self.pool1(enc4)
            enc6 = self.conv3(enc5)
            enc7 = self.act(enc6)
            enc8 = self.conv4(enc7)
            enc9 = self.act(enc8)

            enc10 = self.pool2(enc9)           
            enc11 = self.conv5(enc10)
            enc12 = self.act(enc11)
            enc13 = self.conv6(enc12)
            enc14 = self.act(enc13)
            enc15 = self.conv7(enc14)
            enc16 = self.act(enc15)

            enc17 = self.pool3(enc16)
            enc18 = self.conv8(enc17)
            enc19 = self.act(enc18)
            enc20 = self.conv9(enc19)
            enc21 = self.act(enc20)
            enc22 = self.conv10(enc21)
            enc23 = self.act(enc22)
            # print(enc23.size())

            return enc4, enc9, enc16, enc23


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


        #x16
        # x16 = self.pool1(x16)
        # cont1 = torch.cat((dec3,x16),1)
        # cont1 = self.conv1(cont1)
        # cont1 = self.act(cont1)


        dec4 = self.conv2(dec3)
        dec5 = self.batchNorm2(dec4)
        dec6 = self.act(dec5)
        dec7 = self.conv3(dec6)
        dec8 = self.batchNorm3(dec7)
        dec9 = self.act(dec8)
        dec10 = self.conv4(dec9)
        dec11 = self.batchNorm4(dec10)
        dec12 = self.act(dec11)


        #x9
        # x9 = self.pool1(x9)
        # x9 = self.pool1(x9)
        # cont2 = torch.cat((dec12,x9),1)
        # cont2 = self.conv4(cont2)
        # cont2 = self.act(cont2)


        dec13 = self.conv5(dec12)
        dec14 = self.batchNorm5(dec13)
        dec15 = self.act(dec14)


        #x4
        # x4 = self.pool1(x4)
        # x4 = self.pool1(x4)
        # x4 = self.pool1(x4)
        # cont3 = torch.cat((dec15,x4),1)
        # cont3 = self.conv5(cont3)
        # cont3 = self.act(cont3)


        dec16 = self.conv6(dec15)
        dec17 = self.batchNorm6(dec16)
        dec18 = self.act(dec17)
        # print(dec18.size())

        return dec18


# model = UNet_CANNet2s()
# # model = model.cuda()
# x_prev = torch.randn(2,3,360,640)
# x = torch.randn(2,3,360,640)
# model(x_prev,x)
