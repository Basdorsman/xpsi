#!/bin/bash
#SBATCH -N 1 #5
#SBATCH --tasks-per-node=64
#SBATCH -t 1-00:00:00
#SBATCH -p thin
#SBATCH --job-name=A5intel
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

export compiler='intel' #foss/intel
export atmosphere_type='A'
export n_params='5'
export num_energies='32'
export likelihood='custom' #custom, default
export machine='snellius'
export sampling_params='10'
export integrator='s'

export XPSI_DIR=$HOME/xpsi-bas-fork
export LABEL=${atmosphere_type}${n_params}_s${sampling_params}_e${num_energies}_${compiler}
export STORAGE_DIR=$HOME/outputs/$LABEL/$SLURM_JOB_ID

echo This job $LABEL will go to $STORAGE_DIR.

cd $XPSI_DIR
if [ $compiler == "foss" ]
then
module load foss/2022a
module load SciPy-bundle/2022.05-foss-2022a
module load wrapt/1.15.0-foss-2022a
module load matplotlib/3.5.2-foss-2022a
source $HOME/xpsi-bas-fork/venv_foss/bin/activate
LDSHARED="gcc -shared" CC=gcc python setup.py install --${atmosphere_type}${n_params}Hot
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/multinest/MultiNest_v3.12_CMake/multinest/lib/
elif [ $compiler == "intel" ]
then
module load intel/2022a
module load SciPy-bundle/2022.05-intel-2022a
module load matplotlib/3.5.2-intel-2022a
source $HOME/xpsi-bas-fork/venv_intel/bin/activate
LDSHARED="icc -shared" CC=icc python setup.py install --${atmosphere_type}${n_params}Hot
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/multinest_intel/MultiNest_v3.12_CMake/multinest/lib/
export LD_PRELOAD=/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_def.so.1:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_avx2.so.1:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_core.so:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_intel_lp64.so:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/mkl/2021.2.0/lib/intel64/libmkl_intel_thread.so:/sw/arch/Centos8/EB_production/2021/software/imkl/2021.2.0-iimpi-2021a/compiler/2021.2.0/linux/compiler/lib/intel64_lin/libiomp5.so
export MKL_NUM_THREADS=1
#unset I_MPI_PMI_LIBRARY
#export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1

cp -r $XPSI_DIR/AMXPs/* $TMPDIR/
cd $TMPDIR/inference_run/

echo 'run sample.py'
srun python sample.py > std.out 2> std.err

mkdir $HOME/outputs
mkdir $HOME/outputs/$LABEL
mkdir $STORAGE_DIR

cp std.out std.err $STORAGE_DIR
cp -r run_${LABEL}/ $STORAGE_DIR
