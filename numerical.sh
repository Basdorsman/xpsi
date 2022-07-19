cp xpsi/surface_radiation_field/archive/hot/numerical_5D.pyx  xpsi/surface_radiation_field/hot.pyx
#cp xpsi/surface_radiation_field/archive/elsewhere/numerical.pyx  xpsi/surface_radiation_field/elsewhere.pyx
CC=gcc python setup.py install --user
rm -r /home/bas/anaconda3/envs/xpsi/lib/python2.7/site-packages/xpsi-0.7.10-py2.7-linux-x86_64.egg
mv /home/bas/.local/lib/python2.7/site-packages/xpsi-0.7.10-py2.7-linux-x86_64.egg /home/bas/anaconda3/envs/xpsi/lib/python2.7/site-packages/
