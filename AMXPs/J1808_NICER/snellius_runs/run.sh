export machine=snellius
export compiler=foss
export num_energies=40 #60 #40
export num_leaves=30 # 50 #30
export sqrt_num_cells=50 #90 #50
export num_rays=512
export live_points=20
export max_iter=1
export run_type=test
export bkg=diskline
export support_factor=2
export scenario=2019
export poisson_noise=True
export poisson_seed=42
export fix_mass=True
export sampler=multi
export LABEL=test_analysis


cd $HOME/xpsi-bas-fork/AMXPs/J1808_NICER/
python ST.py
cd $HOME/xpsi-bas-fork/AMXPs/J1808_NICER/snellius_runs/
