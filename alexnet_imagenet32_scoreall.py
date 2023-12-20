import torch
import torchvision.models as models
from alexnet_cifar import *
import functools
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
import numpy as np
import itertools
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--array', type=int)
args = parser.parse_args()
array_index = args.array

filter_list = np.array([64, 192, 384, 256, 256, 4096, 4096]) * 2

divisor = list(np.geomspace(1, 16, 30))

seed_list = [0,1,2]

para_comb = list(itertools.product(seed_list, divisor))
(seed, d) = para_comb[array_index-1]

num_filters = (filter_list / d).astype(int)  # [32, 32, 64, 128, 256]  filter numbers for each laye
device = 'cuda'
model = AlexNet_CIFAR_NoFC(num_filters).to(device)
projection_head = ProjectionHead(input_dim=num_filters[-1], hidden_dim=512, output_dim=128).to(device)

num_para = count_parameters(model)
print("num_parameters:",count_parameters(model))

lr = 3.3e-4

# Define ResNet18
model_ = torch.load(f'/home/mila/z/zixiang.huang/scratch/ps_trash/models/alexnet_simclr_imagenet32_parameters{num_para}_seed{seed}_epoch50_lr{lr}.pt')

preprocessing = functools.partial(load_preprocess_images, image_size=32)

activations_model = PytorchWrapper(identifier=f'alexnet-imagenet32-p{num_para}-s{seed}-lr{lr}', model=model_ , preprocessing=preprocessing)


model = ModelCommitment(identifier=f'alexnet-imagenet32-p{num_para}-s{seed}-lr{lr}', activations_model=activations_model,
                        # specify layers to consider
                        # layers=['avgpool', 'layer3','layer4'])
                        # layers=['features.2', 'features.5', 'features.7', 'features.9', 'features.12', 'classifier.2', 'classifier.5'])
                        layers=['features.9', 'features.12', 'classifier.2', 'classifier.5'])

score = score_model(model_identifier=model.identifier, model=model,
                    benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
print(score)

with open(f'/home/mila/z/zixiang.huang/scratch/ps_trash/brainscore_data/alexnet_imagenet32_p{num_para}_s{seed}_e50_lr{lr}.npy', 'wb') as f:
    np.save(f, score.data)