#!/bin/bash
#SBATCH -A research
#SBATCH -w gnode003
#SBATCH -n 36
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=shreya.patil@research.iiit.ac.in

source /home2/shreya.patil/py-envs/testenv/bin/activate
python /home2/shreya.patil/Courses/NLP/Code-mix/src/dataset_final.py