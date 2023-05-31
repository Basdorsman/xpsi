#!/bin/bash
#SBATCH -N 1 #5
#SBATCH --tasks-per-node=50
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
module load Anaconda3/2022.05
source /sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate xpsi_py3
module load intel/2022a

export atmosphere_type='A'
export n_params='4'
export likelihood='custom' #custom, default


cd $HOME/xpsi-bas-fork/
LDSHARED="icc -shared" CC=icc python setup.py install --${atmosphere_type}${n_params}Hot

cp -r $HOME/xpsi-bas-fork/AMXPs/* $TMPDIR/
cd $TMPDIR/inference_run/
echo 'make tmp snellius runs folder' 
mkdir $TMPDIR/inference_run/snellius_runs/
#mkdir $TMPDIR/inference_run/snellius_runs/run_${atmosphere_type}${n_params}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/multinest/MultiNest_v3.12_CMake/multinest/lib/

export LD_PRELOAD=/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_def.so.1:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_avx2.so.1:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_core.so:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_intel_lp64.so:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_intel_thread.so:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/compiler/2021.2.0/linux/compiler/lib/intel64_lin/libiomp5.so

srun python sample.py > ${atmosphere_type}${n_params}${likelihood}_$SLURM_JOB_ID.out 2> ${atmosphere_type}${n_params}${likelihood}_$SLURM_JOB_ID.err
echo 'make home snellius_runs folder'
mkdir $HOME/xpsi-bas-fork/AMXPs/inference_run/snellius_runs
echo 'make run_atmosphereparams folder'
mkdir $HOME/xpsi-bas-fork/AMXPs/inference_run/snellius_runs/run_${atmosphere_type}${n_params}
cp ${atmosphere_type}${n_params}${likelihood}_$SLURM_JOB_ID.out ${atmosphere_type}${n_params}${likelihood}_$SLURM_JOB_ID.err $HOME/xpsi-bas-fork/AMXPs/inference_run/snellius_runs
cp -r snellius_runs/run_${atmosphere_type}${n_params} $HOME/xpsi-bas-fork/AMXPs/inference_run/snellius_runs
