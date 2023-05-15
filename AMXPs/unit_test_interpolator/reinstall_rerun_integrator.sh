# reinstall integrator and run the python file.
#echo "number of parameters in atmosphere: (choose 4 or 5 for numerical RMP atmosphere, A for accreting, or B for Blackbody)"
#read n_params
n_params=A
cd integrator_stripped
source install_integrator.sh n_params
cd ../
export n_params
python test_integrator_stripped.py
