echo 'load modules, activate environment, reinstall x-psi'

unset LD_LIBRARY_PATH

module purge
module load 2022
module load foss/2022a
#module load OpenSSL/1.1
module load SciPy-bundle/2022.05-foss-2022a
module load wrapt/1.15.0-foss-2022a
module load matplotlib/3.5.2-foss-2022a
source $HOME/xpsi-group/venv_xpsi_group/bin/activate

cd $HOME/xpsi-group/
#rm -r build dist *egg* xpsi/*/*.c xpsi/include/rayXpanda/*.o
LDSHARED="gcc -shared" CC=gcc python setup.py install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/multinest/MultiNest_v3.12_CMake/multinest/lib/

cd $HOME/xpsi-bas-fork/AMXPs/J1808_group/snellius_runs
