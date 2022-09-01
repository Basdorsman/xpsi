# reinstall xpsi and plot_pulses.py in one fell swoop.
echo "number of parameters in atmosphere: (choose 4 or 5)"
read n_params
cd ../../
source numerical.sh n_params
cd tests/plot_pulses/
export n_params
python plot_pulses.py
