import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    '''
    A Custom CNN based on the Very Low early deadline architecture.
    '''

    def __init__(self, outputs=7001, dropout=False):
        """
        Define the layers (convolutional, batchnorm, maxpool, fully connected, etc.)

        Parameters:
            outputs => the number of output classes that the final fully connected layer
                       should map its input to
        """
        super(CustomCNN, self).__init__()
        self.outputs = outputs

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
        if dropout:
            self.fc1 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=1024, out_features=1024),
                nn.ReLU())
            self.fc2 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=1024, out_features=1024),
                nn.ReLU())
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(in_features=1024, out_features=1024),
                nn.ReLU())
            self.fc2 = nn.Sequential(
                nn.Linear(in_features=1024, out_features=1024),
                nn.ReLU())
        self.fc3 = nn.Linear(in_features=1024, out_features=self.outputs)

    def forward(self, x):
        """
        Pass the input through each layer in order.
        Parameters:
            x => Input to the CNN
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
