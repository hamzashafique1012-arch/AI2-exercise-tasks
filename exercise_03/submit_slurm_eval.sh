#!/bin/bash
#SBATCH --job-name=eval_%u
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err
#SBATCH --time=01:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from exercise_03}"

source /opt/miniconda3/bin/activate /opt/conda_envs/course_env

python -u src/evaluate.py --config "${1:-config.yaml}" \
    --checkpoint "${2:-outputs/checkpoints/plantwild_baseline_best.pt}"
