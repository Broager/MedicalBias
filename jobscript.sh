#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J VoxelmorphTest2
### -- ask for number of cores (default: 1) --
#BSUB -n 2
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- Select resoruces: 2 gpu in exclusive mode --
#BSUB -gpu "num=2:mode=exclusive_process"
### -- specify that we need 2GB of memory per core/slot --
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "select[gpu80gb]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot --
#BSUB -M 3GB
### -- set walltime limit: hh:mm --
#BSUB -W 72:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o jobs/Output_%J.out
#BSUB -e jobs/Error_%J.err

# here follow the commands you want to execute
nvidia-smi
module load cuda/11.1
#module load cuda/10.2
source ~venv_1/bin/activate

#python3 my_train_TransMorph.py --loss deepsim-med3d --lam 0.125
python3 DataTest.py

#python3 my_train_TransMorph.py --loss ncc --lam 1