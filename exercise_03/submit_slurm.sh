#!/bin/bash
#SBATCH --job-name=training_%u
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=05:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from exercise_03}"

source /opt/miniconda3/bin/activate /opt/conda_envs/course_env

python -u src/train.py --config "${1:-config.yaml}"
