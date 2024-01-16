echo 'load modules, activate environment, reinstall x-psi'

unset LD_LIBRARY_PATH

module purge
module load 2022
module load foss/2022a
module load SciPy-bundle/2022.05-foss-2022a
module load wrapt/1.15.0-foss-2022a
module load matplotlib/3.5.2-foss-2022a
source $HOME/xpsi-bas-fork/venv_foss/bin/activate

cd $HOME/xpsi-bas-fork
LDSHARED="gcc -shared" CC=gcc python setup.py install --A5Hot
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/multinest/MultiNest_v3.12_CMake/multinest/lib/

cd $HOME/xpsi-bas-fork/AMXPs/J1808_synthetic/snellius_runs
