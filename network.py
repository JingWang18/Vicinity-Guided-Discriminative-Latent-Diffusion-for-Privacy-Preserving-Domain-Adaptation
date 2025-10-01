import torch
import torch.nn as nn

import numpy as np

import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict

import pdb

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50,
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}


class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super().__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        out=x
        return out


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class UNetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, channel_group):
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out, kernel_size=1)
        self.norm = nn.GroupNorm(channel_group, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class FeatureEmbedder(nn.Module):
    def __init__(self, in_features, out_features_list):
        super(FeatureEmbedder, self).__init__()
        self.embeddings = nn.ModuleList([nn.Linear(in_features, out_features) for out_features in out_features_list])

    def forward(self, x):
        return [embedding(x) for embedding in self.embeddings]


class ConditionalUNet(nn.Module):
    def __init__(self):
        super(ConditionalUNet, self).__init__()
        
        # Feature embedder for different levels
        # self.feature_embedder = FeatureEmbedder(1, [64, 128, 256, 512])
        self.feature_embedder = FeatureEmbedder(1, [64, 128, 256])
        
        # U-Net architecture
        self.encoder1 = UNetBlock(dim_in=2048 + 64, dim_out=1024, channel_group=8)
        self.encoder2 = UNetBlock(dim_in=1024 + 128, dim_out=512, channel_group=4)
        self.encoder3 = UNetBlock(dim_in=512 + 256, dim_out=256, channel_group=2)
        # self.encoder3_res = nn.Conv1d(1024, 1024, kernel_size=1)

        self.res = nn.Conv1d(256, 1024, kernel_size=1)

        self.decoder2 = UNetBlock(dim_in=256 + 512, dim_out=512, channel_group=4)
        self.decoder3 = UNetBlock(dim_in=512 + 1024, dim_out=1024, channel_group=2)
        
        self.final_conv = nn.Conv1d(1024, 2048, kernel_size=1)
        
        # BatchNormalization layers
        self.batch_norm1 = nn.BatchNorm1d(1024)
        # self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm_decoder2 = nn.BatchNorm1d(512)

    def forward(self, x, feature):

        x0 = x

        embedded_features = self.feature_embedder(feature)

        # Encoder with feature addition
        x = torch.cat([x, embedded_features[0].unsqueeze(-1)], 1)
        enc1 = self.encoder1(x)
        enc1 = self.batch_norm1(enc1)  # Apply BatchNorm
        
        x = torch.cat([enc1, embedded_features[1].unsqueeze(-1)], 1)
        enc2 = self.encoder2(x)
        # enc2 = self.batch_norm2(enc2)  # Apply BatchNorm
        
        x = torch.cat([enc2, embedded_features[2].unsqueeze(-1)], 1)
        enc3 = self.encoder3(x)
        # enc3 = enc3 + self.res(x0)

        dec2 = self.decoder2(torch.cat([enc3, enc2], 1))
        # dec2 = self.batch_norm_decoder2(dec2)  # Apply BatchNorm

        dec3 = self.decoder3(torch.cat([dec2, enc1], 1))

        out = self.final_conv(dec3)
        return out