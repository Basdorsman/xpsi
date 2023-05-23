#!/bin/bash
#SBATCH -N 1
#SBATCH --tasks-per-node=128
#SBATCH -t 1-00:00:00
#SBATCH -p thin
#SBATCH --job-name=ExampleN4
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

cp -r $HOME/xpsi-bas-fork/examples/examples_modeling_tutorial/* $TMPDIR/
mkdir $TMPDIR/run
cd $TMPDIR/

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/multinest/MultiNest_v3.12_CMake/multinest/lib/

export LD_PRELOAD=/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_def.so.1:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_avx2.so.1:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_core.so:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_intel_lp64.so:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_intel_thread.so:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/compiler/2021.2.0/linux/compiler/lib/intel64_lin/libiomp5.so

srun python TestRun_Num.py > out1 2> err1
cp out1 err1 $HOME/xpsi-bas-fork/examples/examples_modeling_tutorial/.
cp -r run $HOME/xpsi-bas-fork/examples/examples_modeling_tutorial/
#end of job file
