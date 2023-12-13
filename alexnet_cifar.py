import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet_CIFAR_NoFC(nn.Module):
    def __init__(self, num_filters=None):
        super(AlexNet_CIFAR_NoFC, self).__init__()

        # Default number of filters for each convolutional layer if not provided
        if num_filters is None:
            num_filters = [64, 192, 384, 256, 256, 4096, 4096]

        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters[0], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(num_filters[1], num_filters[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters[2], num_filters[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters[3], num_filters[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),        
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=.5),
            nn.Linear(num_filters[4] * 2 * 2, num_filters[5]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.5),
            nn.Linear(num_filters[5], num_filters[6]),
            nn.ReLU(inplace=True),
            # Removed the last FC layer
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
