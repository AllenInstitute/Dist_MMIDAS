#!/bin/bash
#SBATCH -N1
#SBATCH --gpus=v100:4
#SBATCH -c 20
#SBATCH --mem=32G
#SBATCH -p celltypes
#SBATCH -o mnist-logs/mnist_%j.out
#SBATCH -e mnist-logs/mnist_%j.err

source /home/hilal.mufti/miniforge3/etc/profile.d/conda.sh
/home/hilal.mufti/miniforge3/bin/mamba activate mdist-mmidas

python fsdp_mnist.py --epochs 100 --batch-size 256 --model deep --gpus 4 --num_workers 4 --sharding full   --use_batchnorm   --mixed none
