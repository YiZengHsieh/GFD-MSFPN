import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from contextual_layer import ContextualModule

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)

        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 1024
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(256*5, 256, 1, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ScalePyramidModule(nn.Module):
    def __init__(self):
        super(ScalePyramidModule, self).__init__()
        self.assp = ASPP(1024, output_stride=16, BatchNorm=None)
        self.can = ContextualModule(1024, 1024)
        
    def forward(self, *input):
        conv4_3, conv5_3 = input
        
        conv4_3 = self.can(conv4_3)
        conv5_3 = self.assp(conv5_3)
        
        return conv4_3, conv5_3

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vgg = VGG()
        self.spm = ScalePyramidModule()
        # self.range1 = 0.07696
        self.amp = BackEnd()
        self.dmp = BackEnd()


        self.conv_att = BaseConv(64, 1, 1, 1, activation=nn.Sigmoid(), use_bn=True)
        self.conv_out = BaseConv(64, 10, 1, 1, activation=None, use_bn=False)

        self.pool1 = nn.AvgPool2d(kernel_size = 2, stride = 2)

    def forward(self, x_prev, x):
        x_prev2_2, x_prev3_3, x_prev4_3, x_prev5_3 = self.vgg(x_prev)
        x2_2, x3_3, x4_3, x5_3 = self.vgg(x)

        input2_2 = torch.cat([x_prev2_2, x2_2], 1)
        input3_3 = torch.cat([x_prev3_3, x3_3], 1)
        input4_3 = torch.cat([x_prev4_3, x4_3], 1)
        input5_3 = torch.cat([x_prev5_3, x5_3], 1)

        # print(input2_2.size())
        # print(input3_3.size())
        # print(input4_3.size())
        # print(input5_3.size())

        input4_3, input5_3 = self.spm(input4_3, input5_3)
        amp_out = self.amp(input2_2, input3_3, input4_3, input5_3)
        dmp_out = self.dmp(input2_2, input3_3, input4_3, input5_3)

        # print(amp_out.size())
        amp_out = self.conv_att(amp_out)
        # print(amp_out.size())
        # print(dmp_out.size())
        dmp_out = amp_out * dmp_out
        dmp_out = self.pool1(dmp_out)
        dmp_out = self.conv_out(dmp_out)

        # dmp_out = dmp_out.double()
        # dmp_out = torch.where(dmp_out < self.range1, 0.0, dmp_out)

        return dmp_out


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)

        # input = self.pool(conv4_3)
        input = self.conv5_1(input)
        input = self.conv5_2(input)
        conv5_3 = self.conv5_3(input)

        # print(conv2_2.size())
        # print(conv3_3.size())
        # print(conv4_3.size())
        # print(conv5_3.size())

        return conv2_2, conv3_3, conv4_3, conv5_3


class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        
        self.conv1 = BaseConv(1280, 512, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        
        self.conv3 = BaseConv(1280, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7 = BaseConv(128, 64, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.pool1 = nn.AvgPool2d(kernel_size = 2, stride = 2)

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input

        # print(conv2_2.size())
        # print(conv3_3.size())
        # print(conv4_3.size())
        # print(conv5_3.size())

        # input = self.upsample(conv5_3)
        input = conv5_3
        
        input = torch.cat([input, conv4_3], 1)
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.upsample(input)
        
        input = torch.cat([input, conv3_3, self.upsample(conv5_3)], 1)
        input = self.conv3(input)
        input = self.conv4(input)
        input = self.upsample(input)

        input = torch.cat([input, conv2_2], 1)
        input = self.conv5(input)
        input = self.conv6(input)
        input = self.conv7(input)
        input = self.pool1(input)

        return input


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)
        return input

# model = Model()
# # model = model.cuda()
# x = torch.randn(2,3,360,640)
# x_prev = torch.randn(2,3,360,640)
# x1 = model(x_prev, x)
# print(x1.size())