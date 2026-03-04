#!/bin/bash
#SBATCH --job-name=sedd-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=sedd-%j.log

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

torchrun --standalone --nproc_per_node=8 \
    -m scripts.sedd_train \
    --depth=12 \
    --fp8 \
    --run=${WANDB_RUN:-dummy} \
    --save-every=1000 \
    --sample-every=500 \
    --eval-every=250
