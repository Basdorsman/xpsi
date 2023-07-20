# Python 2

#cp xpsi/surface_radiation_field/archive/hot/numerical_${n_params}D.pyx  xpsi/surface_radiation_field/hot.pyx
#cp xpsi/surface_radiation_field/archive/hot/numerical_${n_params}D.pxd  xpsi/surface_radiation_field/hot.pxd
#CC=gcc python setup.py install #--user

# Python 3
# local
CC=gcc python setup.py install --${atmosphere_type}${n_params}Hot
# snellius intel cores
# LDSHARED="icc -shared" CC=icc python setup.py install --${atmosphere_type}${n_params}Hot