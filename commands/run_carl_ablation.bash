#!/bin/bash
#SBATCH --job-name=carl-CMG
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --output=cac.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

# Load conda
source ~/.bashrc
# Or: source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh  # if ~/.bashrc doesn't source conda

# Activate your conda environment
conda activate cac

# === Run CACeriments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --project Exp-CMG --task turtlebot --algo-name carl --disable-CMG-training &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project Exp-CMG --task car --algo-name carl --disable-CMG-training &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project Exp-CMG --task pvtol --algo-name carl --disable-CMG-training &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project Exp-CMG --task neurallander --algo-name carl --disable-CMG-training &

CUDA_VISIBLE_DEVICES=1 python3 main.py --project Exp-CMG --task turtlebot --algo-name carl &
CUDA_VISIBLE_DEVICES=1 python3 main.py --project Exp-CMG --task car --algo-name carl &
CUDA_VISIBLE_DEVICES=1 python3 main.py --project Exp-CMG --task pvtol --algo-name carl &
CUDA_VISIBLE_DEVICES=1 python3 main.py --project Exp-CMG --task neurallander --algo-name carl &


# === Wait for all background jobs to finish ===
wait