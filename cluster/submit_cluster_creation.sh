#!/bin/bash
#SBATCH --job-name=codebert_thesis
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1  
#SBATCH --time=05:00:00
#SBATCH --partition=edu-thesis
#SBATCH --gres=gpu:1  

cd "$SLURM_SUBMIT_DIR"

echo "Job started on $(hostname) at $(date)"

# Activate Python environment (using local venv in the project folder)
source myvenv/bin/activate

echo "Using GPU:"
nvidia-smi

python3 create_new_cluster.py

echo "Job ended at $(date)"
