import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from alexnet_cifar import *
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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


filter_list = np.array([64, 192, 384, 256, 256, 4096, 4096]) * 2

divisor = list(np.geomspace(1, 16, 30))

seed_list = [0,1,2]

lr_list = [1e-4]

mean_score_list = []
parameters_list = []

for d in divisor:
    for lr in lr_list:
        for seed in seed_list:

            num_filters = (filter_list / d).astype(int)

            model = AlexNet_CIFAR_NoFC(num_filters).to(device)

            num_para = count_parameters(model)

            print("num_parameters:", num_para, "seed", seed, "lr", lr)
            try:
                data_mean, data_std = np.load(f'/home/mila/z/zixiang.huang/scratch/ps_trash/brainscore_data/alexnet_imagenet32_p{num_para}_s{seed}_e50_lr{lr}.npy')
                print(data_mean)
            except:
                print('file missing, ignored')
            mean_score_list.append(data_mean)
            parameters_list.append(num_para)

np.save('alexnet_imagenet32_simclr_meanscore_3seeds.npy', np.array(mean_score_list))
np.save('alexnet_imagenet32_simclr_para_3seeds.npy', np.array(parameters_list))
