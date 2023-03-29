import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, features_dim=512):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(16384, 512)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNNLSTM(nn.Module):
    def __init__(self, features_dim=512):
        super(CNNLSTM, self).__init__()
        self.encoder = CNNEncoder(512)
        self.lstm = nn.LSTM(512, 512)

    def forward(self, x, hidden):
        x = self.encoder(x)
        x, hidden = self.lstm(x.unsqueeze(0), hidden)
        return x.squeeze(0), hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, 512), torch.zeros(1, batch_size, 512))