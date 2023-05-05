#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH -w gnode026
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

python -u score_mbert.py > scores_mbert.out
echo "Process Done :)"
