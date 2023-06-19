#!/bin/bash
#SBATCH -N 1 #5
#SBATCH --tasks-per-node=64
#SBATCH -t 01:00:00 #1-00:00:00
#SBATCH -p thin
#SBATCH --job-name=A4test
#SBATCH --mail-user=b.dorsman@uva.nl
#SBATCH --mail-type=END

echo start of job in directory $SLURM_SUBMIT_DIR
echo number of nodes is $SLURM_JOB_NUM_NODES
echo the allocated nodes are:
echo $SLURM_JOB_NODELIST

module purge

module load 2022
module load foss/2022a
module load SciPy-bundle/2022.05-foss-2022a
module load wrapt/1.15.0-foss-2022a
module load matplotlib/3.5.2-foss-2022a

export atmosphere_type='A'
export n_params='4'
export likelihood='custom' #custom, default


cd $HOME/xpsi-bas-fork/
LDSHARED="gcc -shared" CC=gcc python setup.py install --${atmosphere_type}${n_params}Hot

cp -r $HOME/xpsi-bas-fork/AMXPs/* $TMPDIR/
cd $TMPDIR/inference_run/
echo 'make tmp snellius runs folder' 
mkdir $TMPDIR/inference_run/snellius_runs/
#mkdir $TMPDIR/inference_run/snellius_runs/run_${atmosphere_type}${n_params}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
# export MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/multinest/MultiNest_v3.12_CMake/multinest/lib/

srun python sample.py > ${atmosphere_type}${n_params}${likelihood}_$SLURM_JOB_ID.out 2> ${atmosphere_type}${n_params}${likelihood}_$SLURM_JOB_ID.err
echo 'make home snellius_runs folder'
mkdir $HOME/xpsi-bas-fork/AMXPs/inference_run/snellius_runs
echo 'make run_atmosphereparams folder'
mkdir $HOME/xpsi-bas-fork/AMXPs/inference_run/snellius_runs/run_${atmosphere_type}${n_params}
cp ${atmosphere_type}${n_params}${likelihood}_$SLURM_JOB_ID.out ${atmosphere_type}${n_params}${likelihood}_$SLURM_JOB_ID.err $HOME/xpsi-bas-fork/AMXPs/inference_run/snellius_runs
cp -r snellius_runs/run_${atmosphere_type}${n_params} $HOME/xpsi-bas-fork/AMXPs/inference_run/snellius_runs
