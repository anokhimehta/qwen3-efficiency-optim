#!/bin/bash
#SBATCH --job-name=mathvista_baseline
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=mathvista_%j.out
#SBATCH --error=mathvista_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=am16455@nyu.edu

export HF_HOME=/scratch/am16455/hf_cache

cd /scratch/am16455/qwen3-efficiency-optim

singularity exec --nv --overlay /scratch/am16455/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "cd /scratch/am16455/qwen3-efficiency-optim && python3 evaluate_mathvista.py"