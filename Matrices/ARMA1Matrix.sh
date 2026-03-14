#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# Job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --nodes=1  # Number of nodes to use
#SBATCH --cpus-per-task=32
#SBATCH --time=10-0:0:0   # Walltime limit (DD-HH:MM:SS)
#SBATCH --mem=32G
#SBATCH --mail-user=cworman@iastate.edu   # Email address
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --error=job.%J.out
#SBATCH --output=job.%J.err
#SBATCH --job-name="ARMA1Matrix100"

module load r/4.2.2-py310-ly4mhww
module load r-devtools/2.4.5-py310-r42-r2nftew
module load r-foreach/1.5.2-py310-r42-j34b4yu
module load r-mass/7.3-58.1-py310-r42-42nvrln
module load r-snow/0.4-4-py310-r42-2hwwnrd
mkdir -p ./R-packages
Rscript ARMA1Matrix100.R