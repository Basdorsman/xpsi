cd ../../
export atmosphere_type=A
export n_params=5
export machine='local'
export integrator=x
export compiler=foss
export num_energies=60
export num_leaves=50
export sqrt_num_cells=90
export live_points=20
export max_iter=100
export run_type=sample #test

source numerical.sh atmosphere_type n_params
cd AMXPs/J1808_synthetic/
python main.py
