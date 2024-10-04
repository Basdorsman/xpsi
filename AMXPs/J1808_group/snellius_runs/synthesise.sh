echo 'synthesise'

export atmosphere_type=A
export n_params=5
export machine=snellius
export poisson_noise=True
export poisson_seed=42
export scenario=small_r

cd $HOME/xpsi-bas-fork/AMXPs/J1808_group
python synthesise_data.py

cd $HOME/xpsi-bas-fork/AMXPs/J1808_group/snellius_runs

