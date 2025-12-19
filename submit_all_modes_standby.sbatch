#!/bin/bash
#SBATCH -A labi
#SBATCH -p a10
#SBATCH -q standby
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

mkdir -p logs

modes=(constant_bitrate onoff autocorrelated modulated all_multiplexed)
mode="${modes[$SLURM_ARRAY_TASK_ID]}"

# activate your env however you normally do it:
# source ~/routenetvenv/bin/activate
# or: module load ... ; source activate routenetvenv

python mainn.py --mode "$mode"
