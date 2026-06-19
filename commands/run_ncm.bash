#!/bin/bash
#SBATCH --job-name=NCM
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

eval "$(conda shell.bash hook)"
conda activate cac

echo "Using Python from: $(which python3)"
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

# NCM = CV-STEM supervised contraction metric (Neural Contraction Metric). Offline
# CV-STEM solve + regression; controller u = u* - R^{-1} B^T M e.
python3 main.py --project Exp-Contraction --task cartpole --algo-name ncm &
sleep 3
python3 main.py --project Exp-Contraction --task segway --algo-name ncm &
sleep 3
python3 main.py --project Exp-Contraction --task turtlebot --algo-name ncm &
sleep 3
python3 main.py --project Exp-Contraction --task car --algo-name ncm &
sleep 3
python3 main.py --project Exp-Contraction --task pvtol --algo-name ncm &
sleep 3
python3 main.py --project Exp-Contraction --task neurallander --algo-name ncm &
sleep 3
python3 main.py --project Exp-Contraction --task quadrotor --algo-name ncm &

wait
