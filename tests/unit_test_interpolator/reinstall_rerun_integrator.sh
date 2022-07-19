echo "number of parameters in atmosphere: (choose 4 or 5)"
read dimensionality
cd integrator_stripped
source install_integrator.sh dimensionality
cd ../
export dimensionality
python test_integrator_stripped.py
