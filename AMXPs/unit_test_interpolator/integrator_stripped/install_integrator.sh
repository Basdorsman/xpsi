#echo $n_params
cp archive/numerical_${n_params}D.pyx hot.pyx
cp archive/numerical_${n_params}D.pxd hot.pxd

CC=gcc python setup_integrator_stripped.py install --user
# mv /home/bas/Documents/Projects/x-psi/xpsi_compton_slab_atmosphere/integrator_stripped/dist/integrator_stripped-0.0.0-py2.7-linux-x86_64.egg /home/bas/anaconda3/envs/xpsi/lib/python2.7/site-packages/
export n_params
