import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):  # Puoi cambiare il numero di classi
        super(SimpleCNN, self).__init__()
        # Primo blocco convoluzionale
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  # output: 16 x 84 x 96
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 16 x 42 x 48

        # Secondo blocco convoluzionale
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # output: 32 x 42 x 48
        # Dopo MaxPooling: output: 32 x 21 x 24

        # Terzo blocco convoluzionale
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # output: 64 x 21 x 24
        # Dopo MaxPooling: output: 64 x 10 x 12

        # Fully connected layer
        self.fc1 = nn.Linear(in_features=64 * 10 * 12, out_features=128)  # Primo livello fully connected
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)  # Output finale

    def forward(self, x):
        # Passaggio nei layer convoluzionali con ReLU e MaxPooling
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 -> ReLU -> MaxPool
        
        # Flatten per il passaggio al fully connected
        x = x.view(-1, 64 * 10 * 12)  # Flatten (batch_size, features)
        
        # Passaggio nei layer fully connected
        x = F.relu(self.fc1(x))  # Primo fully connected con ReLU
        x = self.fc2(x)  # Output finale
        return x