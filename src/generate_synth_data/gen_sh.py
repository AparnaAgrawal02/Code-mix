

def make_file(n) :
    file_name = f'run_{n}.sh'
    file_content = f"""#!/bin/bash
#SBATCH -A research
#SBATCH -w gnode003
#SBATCH -n 1
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=6G
#SBATCH --time=2-00:00:00
#SBATCH --output=op_file_{n}.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=shreya.patil@research.iiit.ac.in

source /home2/shreya.patil/py-envs/testenv/bin/activate
python /home2/shreya.patil/Courses/NLP/Code-mix/src/dataset_final.py {n}"""
    with open(file_name, 'w') as f :
        f.write(file_content)

for i in range(12) :
    make_file(i)