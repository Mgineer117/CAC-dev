#!/bin/bash
#SBATCH --job-name=CORL
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

# CORL = SD-LQR-pretrained contraction metric + RL on the (running-avg normalized)
# -e^T M e reward. Known dynamics; CMG is pretrained then frozen.
python3 main.py --project Exp-Contraction --task cartpole --algo-name corl &
sleep 3
python3 main.py --project Exp-Contraction --task segway --algo-name corl &
sleep 3
python3 main.py --project Exp-Contraction --task turtlebot --algo-name corl &
sleep 3
python3 main.py --project Exp-Contraction --task car --algo-name corl &
sleep 3
python3 main.py --project Exp-Contraction --task pvtol --algo-name corl &
sleep 3
python3 main.py --project Exp-Contraction --task neurallander --algo-name corl &
sleep 3
python3 main.py --project Exp-Contraction --task quadrotor --algo-name corl &

wait
