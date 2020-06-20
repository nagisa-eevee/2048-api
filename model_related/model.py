import torch
import torch.nn as nn
from torchsummary import summary

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def sequ(kernel_length, input=16, hidden=128, output=128, row_first=False):
    if row_first is True:
        return nn.Sequential(
            nn.Conv2d(input, hidden, kernel_size=(kernel_length, 1), stride=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, output, kernel_size=(1, kernel_length), stride=1),
            nn.BatchNorm2d(output),
            nn.ReLU())
    else:
        return nn.Sequential(
            nn.Conv2d(input, hidden, kernel_size=(1, kernel_length), stride=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, output, kernel_size=(kernel_length, 1), stride=1),
            nn.BatchNorm2d(output),
            nn.ReLU())


channel = 128


# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ConvNet, self).__init__()
        self.layer1 = sequ(2, 16, channel*2, channel)
        self.layer3 = sequ(3, 16, channel*2, channel)
        self.layer5 = sequ(4, 16, channel*2, channel)
        self.dense1 = nn.Sequential(
            nn.Linear(channel * (3*3 + 2*2 + 1*1), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU())
        self.dense2 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out1 = self.layer1(x)
        out3 = self.layer3(x)
        out5 = self.layer5(x)
        out = torch.cat((out1.reshape(out1.size(0), channel, -1),
                         out3.reshape(out3.size(0), channel, -1),
                         out5.reshape(out5.size(0), channel, -1)), dim=2).unsqueeze(dim=2)
        out = out.reshape(out.size(0), -1)
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    model = ConvNet()
    summary(model, input_size=(16, 4, 4), batch_size=-1)
    # for param in model.named_parameters():
    #     print(param[0])
    #     print(param[1].shape)
