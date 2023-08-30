#!/bin/bash
#SBATCH -N 1 #5
#SBATCH --tasks-per-node=64
#SBATCH -t 01:00:00 #1-00:00:00
#SBATCH -p thin
#SBATCH --job-name=N4
##SBATCH --output=/home/dorsman/xpsi-bas-fork/examples/examples_modeling_tutorial/outputs/N4_%j.out
##SBATCH --error=/home/dorsman/xpsi-bas-fork/examples/examples_modeling_tutorial/outputs/N4_%j.err
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

export atmosphere_type="N"
export n_params="4"

cd $HOME/xpsi-bas-fork/
LDSHARED="icc -shared" CC=icc python setup.py install --${atmosphere_type}${n_params}Hot

cp -r $HOME/xpsi-bas-fork/examples/examples_modeling_tutorial/* $TMPDIR/
mkdir $TMPDIR/run
cd $TMPDIR/

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/multinest/MultiNest_v3.12_CMake/multinest/lib/

export LD_PRELOAD=/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_def.so.1:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_avx2.so.1:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_core.so:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_intel_lp64.so:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_intel_thread.so:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/compiler/2021.2.0/linux/compiler/lib/intel64_lin/libiomp5.so

srun python TestRun_Num.py > N4_$SLURM_JOB_ID.out 2> N4_$SLURM_JOB_ID.err
cp N4_$SLURM_JOB_ID.out N4_$SLURM_JOB_ID.err $HOME/xpsi-bas-fork/examples/examples_modeling_tutorial/outputs/
cp -r run $HOME/xpsi-bas-fork/examples/examples_modeling_tutorial/outputs/
#end of job file
