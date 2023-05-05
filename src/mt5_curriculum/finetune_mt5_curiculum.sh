#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
mkdir -p /ssd_scratch/cvit/aparna/
rsync -r aparna@ada.iiit.ac.in:/share3/aparna/mt5_synthetic /ssd_scratch/cvit/aparna/
python finetune_mt5_curiculum.py 
echo "Process Done :)"
