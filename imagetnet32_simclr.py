import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

def get_simclr_data_transforms(input_size=32):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

    data_transforms = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(x)),
        transforms.RandomResizedCrop(size=input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])),
    ])
    return data_transforms


class ImageNet32SimCLRDataset(Dataset):
    def __init__(self, npz_files, transform=None):
        self.images = []
        self.transform = transform

        for npz_file in npz_files:
            data = np.load(npz_file)
            images = data['data']
            # Reshape and append images
            images = images.reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
            self.images.append(images)

        self.images = np.concatenate(self.images, axis=0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)
            return image1, image2
        return image, image

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

filter_list = np.array([64,64,128,256,512])*2

divisor = list(np.geomspace(1, 16, 30))

seed_list = [0,1,2]

para_comb = list(itertools.product(seed_list, divisor))
(seed, d) = para_comb[array_index-1]

num_filters = (filter_list / d).astype(int)  # [32, 32, 64, 128, 256]  filter numbers for each layer
set_seed(seed)

# Define your transformations
transform = get_simclr_data_transforms(input_size=32)

# Paths to your npz files
npz_files = [f'/home/mila/z/zixiang.huang/scratch/ps_trash/imagenet32/Imagenet32_train_npz/train_data_batch_{i}.npz' for i in range(1,11)]
npz_files_val = [f'/home/mila/z/zixiang.huang/scratch/ps_trash/imagenet32/Imagenet32_val_npz/val_data.npz']
# Initialize the dataset
imagenet32_dataset = ImageNet32SimCLRDataset(npz_files=npz_files, transform=transform)
imagenet32_dataset_val = ImageNet32SimCLRDataset(npz_files=npz_files_val, transform=transform)
# Create DataLoader
batch_size = 4096
dataloader = DataLoader(imagenet32_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
batch_size_val = 5000
dataloader_val = DataLoader(imagenet32_dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=12)

num_epochs = 50
model = ResNet18_NoFC(BasicBlock, [2, 2, 2, 2], num_filters).to(device)
projection_head = ProjectionHead(input_dim=num_filters[-1], hidden_dim=512, output_dim=128).to(device)

print("num_parameters:",count_parameters(model))

def train(train_loader, dataloader_val, model, projection_head, optimizer, scheduler, temperature=0.15, epochs=num_epochs):
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

        for (images1, images2) in dataloader_val:
            model.eval()
            with torch.no_grad():
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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2)
model = train(dataloader, dataloader_val, model, projection_head, optimizer, scheduler)

torch.save(model,f"/home/mila/z/zixiang.huang/scratch/ps_trash/models/resnet18_simclr_imagenet32_parameters{count_parameters(model)}_seed{seed}_epoch50.pt")
