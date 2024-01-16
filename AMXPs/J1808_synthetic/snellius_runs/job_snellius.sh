#!/bin/bash
#SBATCH -N 5
#SBATCH --tasks-per-node=192
#SBATCH -t 1-00:00:00
#SBATCH -p genoa
#SBATCH --job-name=mock1808
#SBATCH --mail-user=b.dorsman@uva.nl
#SBATCH --mail-type=END

echo start of job in directory $SLURM_SUBMIT_DIR
echo number of nodes is $SLURM_JOB_NUM_NODES
echo the allocated nodes are:
echo $SLURM_JOB_NODELIST

unset LD_LIBRARY_PATH
#export SLURM_JOB_ID='test_job_id'

module purge
module load 2022

export atmosphere_type=A
export n_params=5
export num_energies=40  # 60
export num_leaves=30  # 50
export sqrt_num_cells=50  # 90
export machine=snellius
export integrator=x
export live_points=1000 #$SLURM_TASKS_PER_NODE
export max_iter=-1
export run_type=sample

export XPSI_DIR=$HOME/xpsi-bas-fork
export LABEL=${SLURM_JOB_NAME}_lp${live_points}
export STORAGE_DIR=$HOME/outputs/$LABEL/$SLURM_JOB_ID

echo This job $LABEL will go to $STORAGE_DIR.

cd $XPSI_DIR
module load foss/2022a
module load SciPy-bundle/2022.05-foss-2022a
module load wrapt/1.15.0-foss-2022a
module load matplotlib/3.5.2-foss-2022a
source $HOME/xpsi-bas-fork/venv_foss/bin/activate
LDSHARED="gcc -shared" CC=gcc python setup.py install --${atmosphere_type}${n_params}Hot
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/multinest/MultiNest_v3.12_CMake/multinest/lib/

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1

cp -r $XPSI_DIR/AMXPs/* $TMPDIR/
cd $TMPDIR/J1808_synthetic/

echo 'run main.py'
# mpirun python main.py >std.out 2> std.err
srun python main.py > std.out 2> std.err

mkdir $HOME/outputs
mkdir $HOME/outputs/$LABEL
mkdir $STORAGE_DIR

cp std.out std.err $STORAGE_DIR
cp -r $LABEL/ $STORAGE_DIR
