#!/bin/bash
# Slurm array job: one task = one benchmark × one compression ratio.
# 4 benchmarks × 7 CRs = 28 tasks (array 0-27).
# task_id = bench_idx * 7 + cr_idx
# Usage: sbatch scripts/run_h2o.sh
#SBATCH --job-name=h2o_sweep
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --array=0-27
#SBATCH --output=h2o_%A_%a.out
#SBATCH --error=h2o_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=am16455@nyu.edu

export HF_HOME=/scratch/am16455/hf_cache

BENCHMARKS=(realworldqa docvqa mathvista mmmu)
CRS=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)
N_CRS=7

BENCH_IDX=$(( SLURM_ARRAY_TASK_ID / N_CRS ))
CR_IDX=$(( SLURM_ARRAY_TASK_ID % N_CRS ))
BENCH=${BENCHMARKS[$BENCH_IDX]}
CR=${CRS[$CR_IDX]}

singularity exec --nv --overlay /scratch/am16455/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "
        cd /scratch/am16455/qwen3-efficiency-optim
        export PYTHONPATH=/scratch/am16455/qwen3-efficiency-optim
        python3 benchmarks/evaluate_${BENCH}.py --method h2o --compression_ratio ${CR}
    "
