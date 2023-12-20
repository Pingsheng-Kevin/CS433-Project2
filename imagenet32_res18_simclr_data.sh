#!/bin/bash

#SBATCH --array=1-48
#SBATCH --output=scaling_law_imagenet.%A.%a.out
#SBATCH --error=scaling_law_imagenet.%A.%a.err
#SBATCH --partition=long                         # Ask for long job
#SBATCH --cpus-per-task=12                                # Ask for 2 CPUs
#SBATCH --gres=gpu:a100l:1 
#SBATCH --mem=60G                                        # Ask for 8 GB of RAM
#SBATCH --time=12:00:00                                   # The job will run for 120h


module load anaconda/3
conda activate ffcv_ssl

python3 ~/scratch/ps_trash/scaling_law/imagetnet32_simclr_res18_data.py --array $SLURM_ARRAY_TASK_ID
