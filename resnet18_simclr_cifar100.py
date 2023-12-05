import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
import numpy as np
from resnet18 import ResNet18_NoFC, ProjectionHead, BasicBlock, count_parameters
import os, random, itertools


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


class SimCLRTransform:
    def __init__(self, size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class CIFAR100SimCLR(Dataset):
    def __init__(self, root='./data', train=True, transform=None):
        self.dataset = CIFAR100(root=root, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img
    
def nt_xent_loss(z_i, z_j, temperature):
    """
    Calculates the NT-Xent loss.
    z_i, z_j are the representations of two augmentations of the same image, 
    and should be normalized.
    """
    batch_size = z_i.size(0)

    z = torch.cat((z_i, z_j), dim=0)
    sim_matrix = torch.exp(torch.mm(z, z.T) / temperature)

    mask = torch.eye(batch_size, dtype=torch.bool).to(z.device)
    mask = mask.repeat(2, 2)
    sim_matrix = sim_matrix.masked_select(~mask).view(2 * batch_size, -1)

    positives = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature).repeat(2)
    negatives = sim_matrix.sum(dim=-1)

    loss = -torch.log(positives / negatives).mean()
    return loss

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--array', type=int)
args = parser.parse_args()
array_index = args.array

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

filter_list = np.array([64,64,128,256,512])

divisor = list(np.geomspace(1, 32, 40))

seed_list = [0,1,2]

para_comb = list(itertools.product(seed_list, divisor))
(seed, d) = para_comb[array_index-1]

num_filters = (filter_list / d).astype(int)  # [32, 32, 64, 128, 256]  filter numbers for each layer
set_seed(seed)

simclr_transform = SimCLRTransform(5) # cifar 
train_dataset = CIFAR100SimCLR(train=True, transform=simclr_transform)
val_dataset =  CIFAR100SimCLR(train=False, transform=simclr_transform)
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=0)
val_loader =  DataLoader(val_dataset, batch_size=10000, shuffle=False, num_workers=0)

num_epochs = 200
model = ResNet18_NoFC(BasicBlock, [2, 2, 2, 2], num_filters).to(device)
projection_head = ProjectionHead(input_dim=num_filters[-1], hidden_dim=512, output_dim=128).to(device)

print("num_parameters:",count_parameters(model))

def train(train_loader, model, projection_head, optimizer, scheduler, temperature=0.15, epochs=num_epochs):
    for epoch in range(epochs):
        for (images1, images2) in train_loader:
            model.train()
            # Concatenate the images from the two augmentations
            images = torch.cat([images1, images2], dim=0)
            images = images.to(device)

            optimizer.zero_grad()

            features = model(images)
            projections = projection_head(features)
            projections = F.normalize(projections, dim=1)

            loss = nt_xent_loss(projections[:len(images)//2], projections[len(images)//2:], temperature)
            
            loss.backward()
            optimizer.step()
            
        # Validation

        for (images1, images2) in val_loader:
            model.eval()
            # Concatenate the images from the two augmentations
            images = torch.cat([images1, images2], dim=0)
            images = images.to(device)

            features = model(images)
            projections = projection_head(features)
            projections = F.normalize(projections, dim=1)

            val_loss = nt_xent_loss(projections[:len(images)//2], projections[len(images)//2:], temperature)
        
        scheduler.step(val_loss)
            
            # Print loss (or log it)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        
    return model

optimizer = torch.optim.AdamW(list(model.parameters()) + list(projection_head.parameters()), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
model = train(train_loader, model, projection_head, optimizer, scheduler)

torch.save(model,f"/home/mila/p/pingsheng.li/scratch/models/resnet18_simclr_cifar100_parameters{count_parameters(model)}_seed{seed}.pt")
