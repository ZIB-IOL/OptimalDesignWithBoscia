#!/bin/bash
#SBATCH --cpus-per-task=4  # No. of cpus to allocate per task
#SBATCH --time=0-12:00:00 # Specify a Time limit in the format days-hrs:min:sec. Use sinfo to see node time limits
#SBATCH --chdir=/home/htc/dhendryc/SCRATCH/research_projects/OptimalDesignWithBoscia/ODWB # Navigate to the working directory where your script lies
#SBATCH --partition=big  # Specify the desired partition, e.g. gpu, cpu or big (GPU is reserved for ML stuff)
## We want to run on htc-cmp[501-532], exclude all others. Must match partition Nodes exactly (partition has 123-139,141-148 not 123-148).
#SBATCH --exclude=htc-cmp[004-008,011-022,024-025,101-121,123-139,141-148]
## for the exclusive run: SBATCH --exclusive
#SBATCH --mail-user=dhendryc@zib.de
#SBATCH --mail-type=FAIL

echo 'Getting node information'
date;hostname;id;pwd

echo 'Activating virtual environment'
source ~/.bashrc  # Load the .bashrc

#echo 'Enabling Internet Access'
export https_proxy=http://squid.zib.de:3128
export http_proxy=http://squid.zib.de:3128

#echo 'Print out the list of GPUs'
#/usr/bin/nvidia-smi

# Print the visible devices
#echo 'Visible devices:', $CUDA_VISIBLE_DEVICES

# Print input 
echo "Criterion: $1"
echo "Mode: $2"
echo "Type: $3"
echo "Dimension: $4"
echo "Integer Data: $5"
echo "Seed: $6"
echo "Option: $7"
echo "N: $8"

echo 'Running script'
bash
julia
export CRITERION=$1
export MODE=$2
export TYPE=$3
export DIMENSION=$4 
export INTEGER_DATA=$5
export SEED=$6
export OPTION=$7
export N=$8
julia --project run_optimal_design.jl &> cb_$1_$2_$4_$3_$5_$6_$7_$8_$SLURM_JOB_ID.txt

# Finish the script
exit 0