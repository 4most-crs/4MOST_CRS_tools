#!/bin/bash
#SBATCH --job-name=BG_CRS_wmag
#SBATCH --partition=general  # Send to ral or cam nodes
#
#SBATCH --cpus-per-task=60  # Number of OMP threads per MPI core
#SBATCH --mem=256G 
#SBATCH --output=%x_%A_%a.log   # Standard output, needs to be on /mnt/ral for ral workers and /mnt/cam for cam workers. %x is job-name and %j is job ID
#SBATCH --error=%x_%A_%a.log


#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.bandi@sussex.ac.uk

#SBATCH --array=1

survey='CRS_BG'
data_tag='r1925_crs_selection'
cat_vars_path='/its/home/bb345/1-research/1-4MOST/3-CRS/1-clustering/1-clustering_v4/catalogue_vars/'

statistics='wmag'
i_random=3

n_jack=36
#! The variable $SLURM_ARRAY_TASK_ID contains the array index for each job.
#! In this example, each job will be passed its index, so each output file will contain a different value
echo "This is job" $SLURM_ARRAY_TASK_ID
cd ..

module purge

source ~/miniconda_setup.sh # This is a script that sets up the conda environment

conda activate /its/home/bb345/.conda/envs/clustering-3.12
conda list

# Run the program
python /its/home/bb345/1-research/1-4MOST/3-CRS/4-crs_tools/1-4MOST_CRS_tools/CRStools/crs_sam_unbinned_star_overdensities.py

