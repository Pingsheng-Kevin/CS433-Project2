#!/bin/bash

#SBATCH --array=1-120
#SBATCH --output=scaling_law_cifar.%A.%a.out
#SBATCH --error=scaling_law_cifar.%A.%a.err
#SBATCH --partition=long                         # Ask for long job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:rtx8000:1 
#SBATCH --mem=24G                                        # Ask for 8 GB of RAM
#SBATCH --time=5:00:00                                   # The job will run for 120h
#SBATCH --mail-user=pingsheng.li@mail.mcgill.ca
#SBATCH --mail-type=ALL

source ~/scaling_law/load_venv.sh

python3 ~/scaling_law/resnet18_cifar_score_all.py --array $SLURM_ARRAY_TASK_ID
