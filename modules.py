import torch
device = torch.device("cuda")
from torch import nn


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=(1, stride, stride), padding=(padding, padding, padding))
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=(1, stride, stride), padding=(padding, padding, padding), output_padding=(0, stride // 2, stride // 2))
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        # print('BasicConv2d', y.shape)
        if self.act_norm:
            y = self.act(self.norm(y))
            # print('BasicConv2d', y.shape)
        return y


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        # print('ConvSC', y.shape)
        return y


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=(stride, stride, stride), padding=(padding, padding, padding), groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        # print('GroupConv2d', y.shape)
        if self.act_norm:
            y = self.activate(self.norm(y))
            # print('GroupConv2d', y.shape)
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv3d(C_in, C_hid, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # print('Inception', x.shape)
        y = 0
        for layer in self.layers:
            y += layer(x)
            # print('Inception', y.shape)
        # print('final-inception', y.shape)
        return y