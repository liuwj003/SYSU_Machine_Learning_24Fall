import torch.nn as nn
import torch.nn.functional as F


class SoftmaxNN(nn.Module):
    def __init__(self, input_size, output_size):
        # super().__init__()
        super(SoftmaxNN, self).__init__()
        # only one fc layer
        self.fc = nn.Linear(input_size, output_size)  # xW+b
        # self.softmax = nn.Softmax(dim=1)  ---> use log_softmax later

    def forward(self, x):
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = F.log_softmax(x, dim=1)
        return x


class MLP2(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = F.log_softmax(x, dim=1)
        return x


class MLP3(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


class MLP4(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP4, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


class CNN(nn.Module):
    """LeNet-5"""
    def __init__(self, output_size):
        super(CNN, self).__init__()
        # 3x32x32 -> 6x28x28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 6x14x14
        # 6x14x14 -> 16x10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 16x5x5
        # Note: should flatten 16x5x5 in forward()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # flatten
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # out = F.relu(self.fc3(x))
        out = F.log_softmax(self.fc3(x), dim=1)
        return out


class CNN2(nn.Module):
    def __init__(self, output_size):
        super(CNN2, self).__init__()
        # 3x32x32 -> 6x28x28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        # 6x28x28 -> 8x26x26
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 8x13x13
        # 8x13x13 -> 16x9x9
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0)
        # 16x9x9 -> 16x5x5
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 16x2x2

        self.fc1 = nn.Linear(16*2*2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        # print(x.size())
        # flatten
        x = x.view(-1, 16*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # out = F.relu(self.fc3(x))
        out = F.log_softmax(self.fc3(x), dim=1)
        return out


class CNN3(nn.Module):
    def __init__(self, output_size):
        super(CNN3, self).__init__()
        # 3x32x32 -> 6x28x28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # 6x14x14
        # 6x14x14 -> 16x10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # 16x5x5
        # Note: should flatten 16x5x5 in forward()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # flatten
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # out = F.relu(self.fc3(x))
        out = F.log_softmax(self.fc3(x), dim=1)
        return out


class CNN4(nn.Module):
    def __init__(self, output_size):
        super(CNN4, self).__init__()
        # 3x32x32 -> 6x28x28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 6x14x14
        # 6x14x14 -> 16x10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # Note: should flatten 16x5x5 in forward()
        self.fc1 = nn.Linear(16*10*10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        # flatten
        x = x.view(-1, 16*10*10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # out = F.relu(self.fc3(x))
        out = F.log_softmax(self.fc3(x), dim=1)
        return out


class CNN5(nn.Module):
    def __init__(self, output_size):
        super(CNN5, self).__init__()
        # 3x32x32 -> 10x28x28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 10x14x14
        # 10x14x14 -> 32x10x10
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 32x5x5
        # Note: should flatten 32x5x5 in forward()
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # flatten
        x = x.view(-1, 32*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # out = F.relu(self.fc3(x))
        out = F.log_softmax(self.fc3(x), dim=1)
        return out
