export machine=snellius
export compiler=foss
export num_energies=40
export num_leaves=30
export sqrt_num_cells=50
export live_points=20
export max_iter=1
export run_type=sample
export bkg=fix
export support_factor=None
export scenario=literature
export poisson_noise=True
export poisson_seed=42


cd $HOME/xpsi-bas-fork/AMXPs/J1808_synthetic/
python run.py
cd $HOME/xpsi-bas-fork/AMXPs/J1808_synthetic/snellius_runs/
