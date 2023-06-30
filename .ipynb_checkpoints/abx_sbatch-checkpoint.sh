#!/bin/bash
# Author(s): James Owers (james.f.owers@gmail.com)
#
# example usage:
# ```
# EXPT_FILE=experiments.txt  # <- this has a command to run on each line
# NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
# MAX_PARALLEL_JOBS=12 
# sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} slurm_arrayjob.sh $EXPT_FILE
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b slurm_arrayjob.sh -e experiments.txt -m 12
# ```


# ====================
# Options for sbatch
# ====================

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/home/%u/aligning-cpc/zerospeech2021/slurm_out/slurm-$1.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/%u/aligning-cpc/zerospeech2021/slurm_out/slurm-$1.out

# Maximum number of nodes to use for the job
# #SBATCH --nodes=2

# Generic resources to use - typically you'll want gpu:n to get n gpus
# #SBATCH --gres=gpu:0

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=14000

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=5

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=4-01:00:00


# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"



# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e


# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
SCRATCH_DISK=/disk/scratch

if [[ "${SLURM_JOB_NODELIST}" = "duflo" ]]
then
    SCRATCH_DISK=/disk/scratch1
fi
echo "scractch disk: ${SCRATCH_DISK}"
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}

# Activate your conda environment
CONDA_ENV_NAME=zerospeech2021
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}


# =================================
# Move input data to scratch disk
# =================================
# Move data from a source location, probably on the distributed filesystem
# (DFS), to the scratch space on the selected node. Your code should read and
# write data on the scratch space attached directly to the compute node (i.e.
# not distributed), *not* the DFS. Writing/reading from the DFS is extremely
# slow because the data must stay consistent on *all* nodes. This constraint
# results in much network traffic and waiting time for you!
#
# This example assumes you have a folder containing all your input data on the
# DFS, and it copies all that data  file to the scratch space, and unzips it. 
#
# For more guidelines about moving files between the distributed filesystem and
# the scratch space on the nodes, see:
#     http://computing.help.inf.ed.ac.uk/cluster-tips

#echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS - change line below if loc different
repo_home=/home/${USER}
src_path=${repo_home}/orthogonal-subspaces

# input data directory path on the scratch disk of the node
dest_path=${SCRATCH_HOME}/orthogonal-subspaces
mkdir -p ${dest_path}/$1  # make it if required

# Important notes about rsync:
# * the --compress option is going to compress the data before transfer to send
#   as a stream. THIS IS IMPORTANT - transferring many files is very very slow
# * the final slash at the end of ${src_path}/ is important if you want to send
#   its contents, rather than the directory itself. For example, without a
#   final slash here, we would create an extra directory at the destination:
#       ${SCRATCH_HOME}/project_name/data/input/input
# * for more about the (endless) rsync options, see the docs:
#       https://download.samba.org/pub/rsync/rsync.html

# if [ -L "${dest_path}/LibriSpeech" ]; then
#   # Take action if $DIR exists. #
#   # ls -l ${dest_path}/LibriSpeech
#   unlink ${dest_path}/LibriSpeech
# fi

rsync --archive --update --compress --progress --exclude='*/' ${src_path}/ ${dest_path}/
rsync --archive --update --compress --progress ${src_path}/$1/ ${dest_path}/$1/
rsync --archive --update --compress --progress ${src_path}/probing_split/ ${dest_path}/probing_split/
rsync --archive --update --compress --progress --exclude='*/' ${src_path}/LibriSpeech/ ${dest_path}/LibriSpeech/
rsync --archive --update --compress --progress --exclude='slurm_out/' ${repo_home}/orthogonal-subspaces/zerospeech2021/ ${SCRATCH_HOME}/orthogonal-subspaces/zerospeech2021


# ==============================
# Finally, run the experiment!
# ==============================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

cd ${dest_path}
cd zerospeech2021
# COMMAND="python -u eval_ABX.py ../$1 ../sterile_split/dev-clean-$2.item --file_extension .npy"
COMMAND="python -u eval_ABX.py ../$1 ABX_data/$2.item --file_extension .npy --feature_size 0.01"

echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"


# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

echo "Moving output data back to DFS"

src_path=${SCRATCH_HOME}/orthogonal-subspaces/$1
dest_path=${repo_home}/orthogonal-subspaces/$1
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}


# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"

