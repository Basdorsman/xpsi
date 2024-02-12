#export atmosphere_type=A
#export n_params=5
export machine='snellius'
#export integrator=x
#export compiler=foss
#export num_energies=60
#export num_leaves=50
#export sqrt_num_cells=90
#export live_points=20
#export max_iter=100
#export run_type=test

cd $HOME/xpsi-bas-fork/AMXPs/J1808_synthetic/
python main.py
