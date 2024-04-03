echo 'synthesise'

export atmosphere_type=A
export n_params=5
export machine=snellius
export poisson_noise=True
export poisson_seed=42

cd $HOME/xpsi-bas-fork/AMXPs/J1808_synthetic
python synthesise_J1808_data.py

cd $HOME/xpsi-bas-fork/AMXPs/J1808_synthetic/snellius_runs
