#!/bin/bash
#SBATCH --cpus-per-task=4  # No. of cpus to allocate per task
#SBATCH --time=0-12:00:00 # Specify a Time limit in the format days-hrs:min:sec. Use sinfo to see node time limits
#SBATCH --chdir=/home/htc/dhendryc/SCRATCH/research_projects/OptimalDesignWithBoscia/ODWB # Navigate to the working directory where your script lies
#SBATCH --partition=big  # Specify the desired partition, e.g. gpu, cpu or big (GPU is reserved for ML stuff)
## Use feature constraints (IT recommendation) for reproducible hardware.
#SBATCH --constraint=Intel&Gold6338&Infiniband
## Request large-memory nodes explicitly.
#SBATCH --mem=900G
## for the exclusive run: SBATCH --exclusive
#SBATCH --mail-user=dhendryc@zib.de
#SBATCH --mail-type=FAIL

echo 'Getting node information'
date;hostname;id;pwd
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE}"
echo "SLURM_MEM_PER_CPU: ${SLURM_MEM_PER_CPU}"

echo 'Hardware summary'
scontrol show node "${SLURMD_NODENAME}" | awk '{for (i=1; i<=NF; i++) if ($i ~ /^(NodeName|Arch|CPUTot|RealMemory|AvailableFeatures)=/) printf "%s ", $i; print ""}'
lscpu | awk -F: '/Model name|CPU MHz|CPU max MHz|CPU min MHz|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core/ {gsub(/^[ \t]+/, "", $2); print $1 ": " $2}'
free -h

echo 'Activating virtual environment'
source ~/.bashrc  # Load the .bashrc

# Force MOSEK runtime/license settings for non-interactive Slurm shells.
export MOSEKBINDIR="/software/mosek/11.1/tools/platform/linux64x86/bin"
export MOSEKLM_LICENSE_FILE="27007@solice01.zib.de"

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
export SEED=$5
export OPTION=$6
export N=$7
LOGFILE="cb_$1_$2_$4_$3_$5_$6_$7_$SLURM_JOB_ID.txt"

# Include runtime resource usage report (CPU time, max RSS, page faults, etc.).
(/usr/bin/time -v julia --project run_optimal_design.jl) &> "${LOGFILE}"

echo 'Accounting summary'
sacct -j "${SLURM_JOB_ID}" --format=JobID,State,Elapsed,TotalCPU,CPUTime,ReqMem,MaxRSS,MaxVMSize,NNodes,NCPUS

# Finish the script
exit 0