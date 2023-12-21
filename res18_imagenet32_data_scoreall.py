import torch
import torchvision.models as models
from resnet18 import *
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

batch_sizes = list(np.geomspace(5000, 1280000, 16).astype(int))  # Customize as needed

seed_list = [0,1,2]

para_comb = list(itertools.product(seed_list, batch_sizes))
(seed, bs) = para_comb[array_index-1]

# Define ResNet18
try:
    model_ = torch.load(f'/home/mila/z/zixiang.huang/scratch/ps_trash/models/resnet18_simclr_imagenet32_seed{seed}_dataset{bs}.pt')
except:
    print("model file missing, ignored.")
    exit(0)

preprocessing = functools.partial(load_preprocess_images, image_size=32)

activations_model = PytorchWrapper(identifier=f'resnet18-imagenet32-ds{bs}-s{seed}', model=model_ , preprocessing=preprocessing)


model = ModelCommitment(identifier=f'resnet18-imagenet32-ds{bs}-s{seed}', activations_model=activations_model,
                        # specify layers to consider
                        # layers=['avgpool', 'layer3','layer4'])
                        layers=['layer3.0.relu','layer3.1.relu','layer4.0.relu','layer4.1.relu', 'avgpool'])

score = score_model(model_identifier=model.identifier, model=model,
                    benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
print(score)

with open(f'/home/mila/z/zixiang.huang/scratch/ps_trash/brainscore_data/resnet18_imagenet32_ds{bs}_s{seed}.npy', 'wb') as f:
    np.save(f, score.data)