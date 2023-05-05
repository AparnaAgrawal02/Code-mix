#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
mkdir -p /ssd_scratch/cvit/aparna/
rsync -r aparna@ada.iiit.ac.in:/share3/aparna/mbert-synthetic /ssd_scratch/cvit/aparna/

python -u score_mbert_synthetic.py >score_mbert_synthetic.out
echo "Process Done :)"
