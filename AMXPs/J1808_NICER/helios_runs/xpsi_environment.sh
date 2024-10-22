module purge
module load gnu12
module load openmpi4
module load gsl
source $HOME/venv311/xpsi/bin/activate
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH
