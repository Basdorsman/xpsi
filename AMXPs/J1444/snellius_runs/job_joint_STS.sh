#!/bin/bash
#SBATCH -N 5
#SBATCH --tasks-per-node=192
#SBATCH -t 5-0:00:00
#SBATCH -p genoa
#SBATCH --job-name=J1444_STS_diskNICER
#SBATCH --mail-user=b.dorsman@uva.nl
#SBATCH --mail-type=END

echo start of job in directory $SLURM_SUBMIT_DIR
echo number of nodes is $SLURM_JOB_NUM_NODES
echo the allocated nodes are:
echo $SLURM_JOB_NODELIST

#unset LD_LIBRARY_PATH

export num_energies=40  # 60
export num_leaves=30  # 50
export sqrt_num_cells=50  # 90
export num_rays=512
export machine=snellius
export live_points=1000 #$SLURM_TASKS_PER_NODE
export max_iter=-1
export run_type=sample
export bkg=disk_NICER
export support_factor=None
export scenario=J1444_STS
export poisson_noise=True
export poisson_seed=42
export sampler=multi
export fix_mass=False
export eos_informed=False
export channel_min=100
export polarization=iqu

export XPSI_DIR=$HOME/xpsi-bas-fork
export LABEL=${SLURM_JOB_NAME}_lp${live_points}
export STORAGE_DIR=$HOME/outputs/$LABEL/$SLURM_JOB_ID
export DIR_NAME=J1444

echo This job $LABEL will go to $STORAGE_DIR.

#cd $HOME/xpsi-group/

module purge
module load 2024
module load foss/2024a
module load SciPy-bundle/2024.05-gfbf-2024a
module load wrapt/1.16.0-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a
module load CMake/3.29.3-GCCcore-13.3.0
module load Cython/3.0.10-GCCcore-13.3.0

source $HOME/venvs/xpsi_py3/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/multinest/MultiNest_v3.12_CMake/multinest/lib/

cp -r $XPSI_DIR/AMXPs/* $TMPDIR/
cd $TMPDIR/$DIR_NAME/
#cp -r $XPSI_DIR/AMXPs/J1444/* $TMPDIR/
#cd $TMPDIR/

echo 'srun python'
srun python joint_STS_variable.py > std.out 2> std.err

mkdir $HOME/outputs
mkdir $HOME/outputs/$LABEL
mkdir $STORAGE_DIR

cp std.out std.err $STORAGE_DIR
cp -r $LABEL/ $STORAGE_DIR

# copy analysis files for posterity
mkdir $STORAGE_DIR/analysis_files

cp $TMPDIR/$DIR_NAME/{joint_STS_variable.py,Custom*,Disk*,parameter_values.py,snellius_runs/job_joint_STS.sh} -r $STORAGE_DIR/analysis_files
