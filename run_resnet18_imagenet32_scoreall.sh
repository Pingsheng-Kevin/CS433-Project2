#!/bin/bash

#SBATCH --array=1-90
#SBATCH --output=scaling_law_imagenet.%A.%a.out
#SBATCH --error=scaling_law_imagenet.%A.%a.err
#SBATCH --partition=long                         # Ask for long job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:a100l:1 
#SBATCH --mem=24G                                        # Ask for 8 GB of RAM
#SBATCH --time=2:00:00                                   # The job will run for 120h


module load anaconda/3
conda activate brainscore

python3 ~/scratch/ps_trash/scaling_law/res18_imagenet32_scoreall.py --array $SLURM_ARRAY_TASK_ID