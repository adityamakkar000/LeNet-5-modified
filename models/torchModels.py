
import torch.nn as nn
import torch.nn.functional as F

"""
PyTorch models for comparison with the custom models

To test use a low lr <0.001
"""
class MLPComparison(nn.Module):

  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(784, 64)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, 10)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    logits = F.tanh(self.fc1(x))
    logits = F.tanh(self.fc2(logits))
    logits = self.fc3(logits)
    x = self.softmax(logits)
    return x, logits

class CNNComparison(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 3, (5,5), stride=1, padding='same')

    self.pool = nn.AvgPool2d((2,2), stride=2)

    self.conv2 = nn.Conv2d(3, 6, (5,5),stride=1, padding='same')


    self.fc1 = nn.Linear(6*7*7, 128)
    self.fc2 = nn.Linear(128, 32)
    self.fc3 = nn.Linear(32, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    self.flattened_x = x.view(x.shape[0], -1)
    x = F.tanh(self.fc1(self.flattened_x))
    x = F.tanh(self.fc2(x))
    logits = self.fc3(x)
    x = F.softmax(logits, dim=1)

    return x, logits