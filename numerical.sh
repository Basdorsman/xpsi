# Python 2

#cp xpsi/surface_radiation_field/archive/hot/numerical_${n_params}D.pyx  xpsi/surface_radiation_field/hot.pyx
#cp xpsi/surface_radiation_field/archive/hot/numerical_${n_params}D.pxd  xpsi/surface_radiation_field/hot.pxd
#CC=gcc python setup.py install #--user

# Python 3

CC=gcc python setup.py install --Num${n_params}DHot