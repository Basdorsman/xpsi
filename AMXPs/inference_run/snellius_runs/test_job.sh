#echo start of job in directory $SLURM_SUBMIT_DIR
#echo number of nodes is $SLURM_JOB_NUM_NODES
#echo the allocated nodes are:
#echo $SLURM_JOB_NODELIST


unset LD_LIBRARY_PATH
export SLURM_JOB_ID='test_job_id'

module purge
module load 2022

export compiler='foss' #'foss'
export atmosphere_type='A'
export n_params=5
export num_energies=40
export num_leaves=30
export sqrt_num_cells=50
export likelihood='custom' #custom, default
export machine='snellius'
export max_iter=100
export sampling_params=10
export live_points=64
export integrator='x'

export XPSI_DIR=$HOME/xpsi-bas-fork
export LABEL=${atmosphere_type}${n_params}_s${sampling_params}_e${num_energies}_${compiler}
export STORAGE_DIR=$HOME/outputs/$LABEL/$SLURM_JOB_ID


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
export MKL_NUM_THREADS=1
unset I_MPI_PMI_LIBRARY
export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1


cp -r $XPSI_DIR/AMXPs/* $TMPDIR/
cd $TMPDIR/inference_run/

echo 'run sample.py'
python sample.py > $SLURM_JOB_ID.out 2> $SLURM_JOB_ID.err

mkdir $HOME/outputs
mkdir $HOME/outputs/$LABEL
mkdir $STORAGE_DIR

cp $SLURM_JOB_ID.out $SLURM_JOB_ID.err $STORAGE_DIR
cp -r run_${LABEL}/ $STORAGE_DIR
