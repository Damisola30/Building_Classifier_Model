import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # First convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # Max pooling
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Second convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # Max pooling
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # Third convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)        # Max pooling
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 50 * 37, 256),                # Fully connected layer
            nn.ReLU(),
            nn.Linear(256, 3)                            # Output layer
        )

    def forward(self, x):
        x = self.conv_layer(x)  # Apply convolutional layers
        x = torch.flatten(x, 1) # Flatten the output for the dense layer
        x = self.fc_layer(x)    # Apply fully connected layers
        return x


