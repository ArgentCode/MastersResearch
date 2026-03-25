#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# Job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --nodes=2   # Number of nodes to use
#SBATCH --ntasks-per-node=6   # Use 32 processor cores per node 
#SBATCH --time=0-8:0:0   # Walltime limit (DD-HH:MM:SS)
#SBATCH --gres=gpu:a100:2   # Required GPU hardware
#SBATCH --mem=32G
#SBATCH --mail-user=cworman@iastate.edu   # Email address
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --chdir=/Lustre/LAS/sabzikar-lab/cworman/MastersResearch
#SBATCH --job-name="Testing"
#SBATCH --error=job-errors-%j.out
#SBATCH --output=job-output-%j.out


Rscript ARMA2.R
