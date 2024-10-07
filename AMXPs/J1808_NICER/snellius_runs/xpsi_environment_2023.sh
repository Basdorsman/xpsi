echo 'load modules, activate environment, reinstall x-psi'

deactivate
unset LD_LIBRARY_PATH

module purge
module load 2023 #2022
module load foss/2023a #foss/2022a
module load SciPy-bundle/2023.07-gfbf-2023a #SciPy-bundle/2022.05-foss-2022a
module load wrapt/1.15.0-gfbf-2023a  #wrapt/1.15.0-foss-2022a
module load matplotlib/3.7.2-gfbf-2023a #matplotlib/3.5.2-foss-2022a
#module load 2022
#module load CMake/3.23.1-GCCcore-11.3.0
#module load CMake/3.26.3-GCCcore-12.3.0
source $HOME/xpsi-group/venv_xpsi_group_2023/bin/activate

#pip install cython==0.29.28
#export PYTHONPATH=$HOME/xpsi-group/venv_xpsi_group_2023/lib/python3.11/site-packages/:$PYTHONPATH

cd $HOME/xpsi-group/
#rm -r build dist *egg* xpsi/*/*.c xpsi/include/rayXpanda/*.o
LDSHARED="gcc -shared" CC=gcc python setup.py install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/multinest/MultiNest_v3.12_CMake/multinest/lib/

cd $HOME/xpsi-bas-fork/AMXPs/J1808_NICER/snellius_runs
