#!/bin/bash

#SBATCH --job-name=matching_objects
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000


# Singularity path
ext3_path=/scratch/lg3490/tfv/overlay-25GB-500K.ext3
sif_path=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
conda activate vis
python main.py
"