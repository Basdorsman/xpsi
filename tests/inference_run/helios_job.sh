#!/bin/bash
#SBATCH -N 1
#SBATCH --tasks-per-node=126
#SBATCH -t 1-00:00:00
#SBATCH -J time_likelihood
#SBATCH -o output/time_likelihood_%j.out
#SBATCH -e output/time_likelihood_%j.err
#SBATCH --partition=neutron-star
#SBATCH --mem 160000
#SBATCH --mail-user=b.dorsman@uva.nl
#SBATCH --mail-type=END

module purge
module load anaconda3/2021-05
conda activate xpsi_py3
module load openmpi/3.1.6

export atmosphere_type="A"
export n_params="5"

export JOBNAME='time_likelihood'
export RUN="run_${atmosphere_type}${n_params}" 
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=$HOME/multinest/MultiNest_v3.12_CMake/multinest/lib/:$LD_LIBRARY_PATH
export PATH=$HOME/gsl/bin:$PATH

export JOB_DIR=$HOME/xpsi-bas/tests/inference_run
export OUTPUT_FOLDER=$(mktemp -d -p /hddstore/$USER)

# INSTALL X-PSI
cd $HOME/xpsi-bas
if [ $atmosphere_type == "N" ] || [ $atmosphere_type == "A" ]
then
source numerical.sh atmosphere_type n_params
elif [ $atmosphere_type == "B" ]
then
source blackbody.sh
fi

echo $OUTPUT_FOLDER $SLURMD_NODENAME
mkdir $OUTPUT_FOLDER/$RUN
cd $OUTPUT_FOLDER

#Copy the input data to be visible for all the nodes (and make sure your paths point to hddstore):
srun -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1 cp -r $JOB_DIR/model_data $OUTPUT_FOLDER
sleep 1

mpiexec -n 126 -mca btl_tcp_if_include ib0 python $JOB_DIR/sample.py

#Move your output from scratch to storage space.
mkdir -p /zfs/helios/filer0/$USER/$JOBNAME/$RUN/$SLURM_JOB_ID
cp -r $OUTPUT_FOLDER/$RUN /zfs/helios/filer0/$USER/$JOBNAME/$RUN/$SLURM_JOB_ID
cp $JOB_DIR/output/time_likelihood_${SLURM_JOB_ID}.* /zfs/helios/filer0/$USER/$JOBNAME/$RUN/$SLURM_JOB_ID

#Clean the scratch automatically here.
#But remember to remove manually in each node, if the main program ends by crashing.
rm -rf $OUTPUT_FOLDER
