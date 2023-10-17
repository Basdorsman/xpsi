# reinstall xpsi and plot_pulses.py in one fell swoop.

# echo "atmosphere type: N for NSX, A for accreting, B for blackbody"
# read atmosphere_type
# echo "number of parameters in atmosphere: choose 4 or 5 for NSX or accreting atmosphere"
# read n_params
# echo "which integrator to invoke: a for azimuthal_invariance_combined, x for azimuthal_invariance_split, c for general_combined, s for general_split, g for general_split_gsl."
# read integrator
atmosphere_type='A'


cd ../../

if [ $atmosphere_type == "N" ] || [ $atmosphere_type == "A" ]
then
source numerical.sh atmosphere_type n_params
elif [ $n_params == "B" ]
then
source blackbody.sh
fi


cd AMXPs/inference_run/
export atmosphere_type
python sample.py ##> test.txt
