#!/bin/bash
#SBATCH -N 1
#SBATCH --tasks-per-node=126
#SBATCH -t 1-00:00:00
#SBATCH -J example_run
#SBATCH -o example_run_%j.out
#SBATCH -e example_run_%j.err
#SBATCH --partition=neutron-star
#SBATCH --mem 160000
#SBATCH --mail-user=
#SBATCH --mail-type=END

module purge
module load anaconda3/2021-05
conda activate xpsi_py3
module load openmpi/3.1.6

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=$HOME/multinest/MultiNest_v3.12_CMake/multinest/lib/:$LD_LIBRARY_PATH
export PATH=$HOME/gsl/bin:$PATH

export JOB_DIR=$HOME/xpsi/tests/inference_run
export OUTPUT_FOLDER=$(mktemp -d -p /hddstore/$USER)
echo $OUTPUT_FOLDER $SLURMD_NODENAME
mkdir $OUTPUT_FOLDER/run
cd $OUTPUT_FOLDER

#Copy the input data to be visible for all the nodes (and make sure your paths point to hddstore):
srun -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1 cp -r $JOB_DIR/model_data $OUTPUT_FOLDER
sleep 1

mpiexec -n 126 -mca btl_tcp_if_include ib0 python $JOB_DIR/sample.py

#Move your output from scratch to storage space.
mkdir -p /zfs/helios/filer0/$USER/
cp -r $OUTPUT_FOLDER/* /zfs/helios/filer0/$USER/

#Clean the scratch automatically here.
#But remember to remove manually in each node, if the main program ends by crashing.
#rm -rf $OUTPUT_FOLDER
