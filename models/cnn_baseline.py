import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, num_classes=7001, dropout=True):
        super(Network, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=4, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)

        # Adaptive average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x, return_feats=False):
        # Convolutional layers with BatchNorm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Adaptive average pooling
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers
        feats = F.relu(self.fc1(x))
        out = self.fc2(feats)

        if return_feats:
            return feats
        else:
            return out

# # Instantiate the model and print its summary
# model = Network().to(DEVICE)
# summary(model, (3, 224, 224))
