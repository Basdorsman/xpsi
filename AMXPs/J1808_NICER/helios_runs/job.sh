#!/bin/bash
#SBATCH -N 4
#SBATCH --tasks-per-node=126
#SBATCH -t 2-00:00:00
#SBATCH -J 2019_disk
#SBATCH --partition=neutron-star
#SBATCH --mem 0
#SBATCH --mail-user=b.dorsman@uva.nl
#SBATCH --mail-type=END

module purge
module load gnu12
module load openmpi4
module load gsl
source $HOME/venv311/xpsi/bin/activate
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH

echo start of job in directory $SLURM_SUBMIT_DIR
echo number of nodes is $SLURM_JOB_NUM_NODES
echo the allocated nodes are $SLURM_JOB_NODELIST

export num_energies=40  # 60
export num_leaves=30  # 50
export sqrt_num_cells=50  # 90
export num_rays=512
export machine=helios
export live_points=1000
export max_iter=-1
export run_type=sample
export bkg=disk
export support_factor=None
export scenario=2019
export sampler=multi

export XPSI_DIR=$HOME/xpsi-bas-fork
export LABEL=${SLURM_JOB_NAME}_lp${live_points}
export STORAGE_DIR=$HOME/outputs/$LABEL/$SLURM_JOB_ID

export TMPDIR=$(mktemp -d /zfs/helios/filer0/bdorsma/xpsi/tmpdir.XXXXXX)
cp -r $XPSI_DIR/AMXPs/* $TMPDIR/

echo This job $LABEL will run in $TMPDIR and then be copied to $STORAGE_DIR.


cd $TMPDIR/J1808_NICER/
total_processes=$(( $(printf "%d" $SLURM_TASKS_PER_NODE) * $(printf "%d" $SLURM_NNODES) ))
echo Running mpirun python with $total_processes processes.

mpirun -n $total_processes python ST.py > std.out 2> std.err

echo If no directories yet, create them:
mkdir $HOME/outputs/$LABEL
mkdir $STORAGE_DIR

echo Copy result contents to $STORAGE_DIR:
cp std.out std.err $STORAGE_DIR
cp -r $LABEL/ $STORAGE_DIR

echo Copy analysis files for posterity
mkdir $STORAGE_DIR/analysis_files
cp $TMPDIR/J1808_NICER/ST.py $STORAGE_DIR/analysis_files
cp $TMPDIR/parameter_values.py $STORAGE_DIR/analysis_files
cp $TMPDIR/J1808_NICER/Custom* $STORAGE_DIR/analysis_files
cp $TMPDIR/J1808_NICER/synthesise_data.py $STORAGE_DIR/analysis_files
cp $TMPDIR/J1808_NICER/helios_runs/job* $STORAGE_DIR/analysis_files
cp -r $TMPDIR/J1808_NICER/data $STORAGE_DIR/analysis_files

