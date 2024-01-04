#!/bin/sh
### General options
### –- specify queue --
BSUB -q gpuv100
### -- set the job Name --
BSUB -J firstJob
### -- ask for number of cores (default: 1) --
BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
BSUB -gpu "num=2:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
BSUB -W 1:00
# request 5GB of system-memory
BSUB -R "rusage[mem=5GB]"
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery