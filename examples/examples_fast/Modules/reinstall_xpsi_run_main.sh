cd ../../../
CC=gcc python setup.py install
cd examples/examples_fast/Modules/
#python main.py
mpirun -np 4 python main.py
