import torch.nn as nn
import torch.nn.init as init


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, use_mhsa=False):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.use_mhsa = use_mhsa
        if use_mhsa:
            self.conv2_1x1_1 = nn.Conv2d(mid_channels, 512, kernel_size=1, bias=False)
            self.mhsa = nn.MultiheadAttention(embed_dim=512, num_heads=8)
            self.conv2_1x1_2 = nn.Conv2d(512, mid_channels, kernel_size=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        # First layer
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        # Second layer
        if self.use_mhsa:
            out = self.conv2_1x1_1(out)
            B, C, H, W = out.size()
            out = out.permute(0, 2, 3, 1).reshape(B * H * W, C)
            out, _ = self.mhsa(out, out, out)
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
            out = self.conv2_1x1_2(out)
        else:
            out = nn.ReLU()(self.bn2(self.conv2(out)))
        # Third layer
        out = self.bn3(self.conv3(out))
        # Down sampling
        out += self.shortcut(identity)
        out = nn.ReLU()(out)

        return out


class ResNetMHSA(nn.Module):
    def __init__(self, num_classes=7001):
        super(ResNetMHSA, self).__init__()

        # Initial Convolution and Max-Pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(64, 64, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 512, 5, stride=2)
        self.layer3 = self._make_layer(512, 256, 1024, 5, stride=2)
        self.layer4 = self._make_layer(1024, 512, 1024, 3, stride=2, use_mhsa=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, in_channels, mid_channels, out_channels, num_blocks, stride, use_mhsa=False):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)  # First block with specified stride, others with stride 1
        for stride in strides:
            layers.append(ResidualBlock(in_channels, mid_channels, out_channels, stride, use_mhsa))
            in_channels = out_channels  # Update in_channels for next block
        return nn.Sequential(*layers)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                init.xavier_uniform_(m.in_proj_weight)
                init.xavier_uniform_(m.out_proj.weight)
                if m.in_proj_bias is not None:
                    init.constant_(m.in_proj_bias, 0)
                if m.out_proj.bias is not None:
                    init.constant_(m.out_proj.bias, 0)
