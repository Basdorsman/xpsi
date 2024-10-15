export machine=snellius
export compiler=foss
export num_energies=40 #60 #40
export num_leaves=30 # 50 #30
export sqrt_num_cells=50 #90 #50
export num_rays=512
export live_points=20
export max_iter=2
export run_type=sample
export bkg=model
export support_factor=None
export scenario=2019
export poisson_noise=True
export poisson_seed=42
export LABEL=2022_lp400


cd $HOME/xpsi-bas-fork/AMXPs/J1808_group/
python analysis.py
cd $HOME/xpsi-bas-fork/AMXPs/J1808_group/snellius_runs/