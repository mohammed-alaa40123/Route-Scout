#!/bin/bash
#SBATCH --job-name=rn-predict-tm
#SBATCH -A standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40GB
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --constraint=K
#SBATCH --output=./Routenet_logs/%x_%A_%a_%j.out
#SBATCH --error=./Routenet_logs/%x_%A_%a_%j.err

# You can submit as an array (see run_samples_jobs.sh)

# Allow SAMPLE_INDEX to be passed via environment or via SLURM_ARRAY_TASK_ID
SAMPLE_INDEX=${SAMPLE_INDEX:-${SLURM_ARRAY_TASK_ID}}
if [ -z "$SAMPLE_INDEX" ]; then
  echo "No SAMPLE_INDEX provided and SLURM_ARRAY_TASK_ID empty. Exiting."
  exit 1
fi

echo "Starting job for SAMPLE_INDEX=$SAMPLE_INDEX"

source /etc/profile.d/modules.sh
source /scratch/gilbreth/mohame43/Routenet/routenet/bin/activat e

# Run the TrafficModels prediction script (it uses SAMPLE_INDEX env variable)
bash /scratch/gilbreth/mohame43/Routenet/predict_with_candidate_routes_scheduling.sh

echo "Job finished for SAMPLE_INDEX=$SAMPLE_INDEX"