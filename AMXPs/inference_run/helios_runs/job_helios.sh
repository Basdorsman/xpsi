#!/bin/bash
#SBATCH -N 1
#SBATCH --tasks-per-node=64
#SBATCH -t 10:00:00
#SBATCH -J A4A5extraparameter
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --partition=neutron-star
#SBATCH --mem 160000
#SBATCH --mail-user=b.dorsman@uva.nl
#SBATCH --mail-type=END

module purge
module load anaconda3/2021-05
conda activate xpsi_py3
module load openmpi/3.1.6

export atmosphere_type='N'
export n_params='4'
export num_energies='16'
export likelihood='custom' #custom, default
export machine='helios'

export JOBNAME=$SLURM_JOB_NAME
export RUN="run_${atmosphere_type}${n_params}${likelihood}" 
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=$HOME/multinest/MultiNest_v3.12_CMake/multinest/lib/:$LD_LIBRARY_PATH
export PATH=$HOME/gsl/bin:$PATH

export XPSI_DIR=$HOME/xpsi-bas
export TMP_DIR=$(mktemp -d -p /hddstore/$USER)

# INSTALL X-PSI
cd $XPSI_DIR
if [ $atmosphere_type == "N" ] || [ $atmosphere_type == "A" ]
then
source numerical.sh atmosphere_type n_params
elif [ $atmosphere_type == "B" ]
then
source blackbody.sh
fi

echo TMP_DIR: ${TMP_DIR}. SLURMD_NODENAME: ${SLURMD_NODENAME}.
mkdir $TMP_DIR/AMXPs

#Copy the input data to be visible for all the nodes (and make sure your paths point to hddstore):
srun -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1 cp -r $XPSI_DIR/AMXPs/* $TMP_DIR/AMXPs
sleep 1

export OUTPUT_FOLDER=$TMP_DIR/AMXPs/inference_run
mkdir $OUTPUT_FOLDER/helios_runs
mkdir $OUTPUT_FOLDER/helios_runs/$RUN

cd $OUTPUT_FOLDER

mpiexec -n 64 -mca btl_tcp_if_include ib0 python $XPSI_DIR/AMXPs/inference_run/sample.py

#Move your output from scratch to storage space.
export STORAGE_DIR=/zfs/helios/filer0/$USER/$JOBNAME/$RUN/$SLURM_JOB_ID
mkdir -p $STORAGE_DIR
cp -r $OUTPUT_FOLDER/helios_runs/$RUN $STORAGE_DIR

#Clean the scratch automatically here.
#But remember to remove manually in each node, if the main program ends by crashing.
rm -rf $TMP_DIR
