#!/bin/bash

#SBATCH --job-name=count_num_objects
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH --time=60:00:00


# Singularity path
ext3_path=/scratch/lg3490/tfv/overlay-25GB-500K.ext3
sif_path=/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif
# start running
singularity exec \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
conda activate vis
python main.py
"