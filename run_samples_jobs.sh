#!/bin/bash -e
# Submit per-sample jobs (SLURM array or individual) for TrafficModels prediction
# Usage:
#   ./run_samples_jobs.sh [start] [end]
# Defaults: start=1 end=20

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
JOB_SCRIPT="$SCRIPT_DIR/sample_job.sbatch"
START=${1:-1}
END=${2:-20}

# Submit as a single array job
echo "Submitting SLURM array job for samples $START..$END"
sbatch --array=${START}-${END} "$JOB_SCRIPT" | tee "$SCRIPT_DIR/submit_array.out"

echo "Submitted. See submit_array.out for the sbatch output."
