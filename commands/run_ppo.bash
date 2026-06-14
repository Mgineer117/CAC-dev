#!/bin/bash
#SBATCH --job-name=PPO
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=cac.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

# 1. The foolproof way to load Conda in a Slurm script
# This directly hooks Conda into the shell without relying on .bashrc
eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate cac

# (Optional but recommended) Load the CUDA module if Illinois Computes requires it
# module load cuda/11.8  # Uncomment and adjust version if you still get errors

echo "Using Python from: $(which python3)"
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

# === Run Experiments in Parallel === # 
# Removed the hardcoded CUDA variables. Slurm handles this for you.
# Added a 3-second sleep between launches to prevent driver timeout during PyTorch initialization.

python3 main.py --project Exp-Contraction --task cartpole --algo-name ppo &
sleep 3
python3 main.py --project Exp-Contraction --task segway --algo-name ppo &
sleep 3
python3 main.py --project Exp-Contraction --task turtlebot --algo-name ppo &
sleep 3
python3 main.py --project Exp-Contraction --task car --algo-name ppo &
sleep 3
python3 main.py --project Exp-Contraction --task pvtol --algo-name ppo &
sleep 3
python3 main.py --project Exp-Contraction --task neurallander --algo-name ppo &
sleep 3
python3 main.py --project Exp-Contraction --task quadrotor --algo-name ppo &

# === Wait for all background jobs to finish ===
wait