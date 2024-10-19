import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, conv1_ch=10, conv2_ch=40, conv3_ch=40, kernel_size=5,
                 fc1_ft=4000, fc2_ft=500, classes=10):
        super().__init__()
        self.conv1 = nn.LazyConv2d(conv1_ch, kernel_size) #count(W*H), kernel
        self.pool = nn.MaxPool2d(2, 2) #was 28 by 28 become 14 by 14
        self.conv2 = nn.LazyConv2d(conv2_ch, kernel_size)
        self.conv3 = nn.LazyConv2d(conv3_ch, kernel_size)
        self.fc1 = nn.LazyLinear(fc1_ft) # 5*5 its size of image after 2 MaxPooling
        self.fc2 = nn.LazyLinear(fc2_ft) 
        self.fc3 = nn.LazyLinear(classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x