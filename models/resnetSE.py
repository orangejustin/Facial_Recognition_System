import torch.nn as nn
import torch
import torch.nn.init as init
from torchsummary import summary

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResNetBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(SEResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = nn.GELU()(self.bn1(self.conv1(x)))
        out = nn.GELU()(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += self.shortcut(identity)
        out = nn.GELU()(out)
        return out


class CustomSEResNet(nn.Module):
    def __init__(self, num_classes=7001, dropout=False, dropout_prob=0.49):
        super(CustomSEResNet, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 512, 3, stride=2)
        self.layer3 = self._make_layer(512, 256, 1024, 3, stride=2)
        self.layer4 = self._make_layer(1024, 512, 1024, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
        if dropout:
            self.dropout_layer = nn.Dropout(dropout_prob)

    def _make_layer(self, in_channels, mid_channels, out_channels, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(SEResNetBlock(in_channels, mid_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, return_feats=False):
        x = nn.GELU()(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if return_feats:
            return x
        if self.dropout:
            x = self.dropout_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CustomSEResNet(dropout=True, dropout_prob=0.3).to(DEVICE)
# model.initialize_weights()
# print(summary(model, (3, 224, 224)))