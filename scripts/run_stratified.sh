#!/bin/bash
# Slurm array job: one task = one benchmark × one (compression_ratio, vision_weight) pair.
# 4 benchmarks × 12 (CR, VW) combos = 48 tasks (array 0-47).
# task_id = bench_idx * 12 + param_idx
# Usage: sbatch scripts/run_stratified.sh
#SBATCH --job-name=stratified_sweep
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --array=0-47
#SBATCH --output=stratified_%A_%a.out
#SBATCH --error=stratified_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=am16455@nyu.edu

export HF_HOME=/scratch/am16455/hf_cache

BENCHMARKS=(realworldqa docvqa mathvista mmmu)

# 3 CRs × 4 VWs = 12 (CR, VW) pairs
CRS=(0.3 0.3 0.3 0.3  0.5 0.5 0.5 0.5  0.7 0.7 0.7 0.7)
VWS=(0.1 0.2 0.3 0.5  0.1 0.2 0.3 0.5  0.1 0.2 0.3 0.5)
N_PARAMS=12

BENCH_IDX=$(( SLURM_ARRAY_TASK_ID / N_PARAMS ))
PARAM_IDX=$(( SLURM_ARRAY_TASK_ID % N_PARAMS ))
BENCH=${BENCHMARKS[$BENCH_IDX]}
CR=${CRS[$PARAM_IDX]}
VW=${VWS[$PARAM_IDX]}

singularity exec --nv --overlay /scratch/am16455/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "
        cd /scratch/am16455/qwen3-efficiency-optim
        export PYTHONPATH=/scratch/am16455/qwen3-efficiency-optim
        python3 benchmarks/evaluate_${BENCH}.py --method stratified_eviction --compression_ratio ${CR} --vision_weight ${VW}
    "
