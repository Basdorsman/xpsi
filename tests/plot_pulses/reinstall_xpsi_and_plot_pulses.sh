# reinstall xpsi and plot_pulses.py in one fell swoop.
echo "number of parameters in atmosphere: (choose 4 or 5 for numerical RMP atmosphere, A for accreting, or B for Blackbody)"
read n_params
cd ../../

if [ $n_params == "4" ] || [ $n_params == "5" ] || [ $n_params == "A" ]
then
source numerical.sh n_params
elif [ $n_params == "B" ]
then
source blackbody.sh
fi

cd tests/plot_pulses/
export n_params
python plot_pulses.py
