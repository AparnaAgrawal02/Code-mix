#!/bin/bash
#SBATCH -A aparna
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
python finetune_mbert.py 
echo "Process Done :)"
