import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=8):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):

    def __init__(self, channels, height, width, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 8, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, stride=2, padding=2)
        self.num_flat_features = 8 * (height // 4) * (width // 4)
        self.fc1 = nn.Linear(self.num_flat_features, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.view(-1, self.num_flat_features)))
        x = self.fc2(x)
        return x
