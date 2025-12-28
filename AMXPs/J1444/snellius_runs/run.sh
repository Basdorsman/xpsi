export machine=snellius
export num_energies=40 #60 #40
export num_leaves=30 # 50 #30
export sqrt_num_cells=50 #90 #50
export num_rays=512
export live_points=20
export max_iter=1
export run_type=sample
export bkg=disk
export support_factor=None
export scenario=J1444_STS
export poisson_noise=True
export poisson_seed=42
export fix_mass=False
export sampler=multi
export LABEL=test_analysis
export eos_informed=False
export polarization=iqu
export channel_min=100

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/multinest/MultiNest_v3.12_CMake/multinest/lib/

cd $HOME/xpsi-bas-fork/AMXPs/J1444/
python joint_STS_variable.py
cd $HOME/xpsi-bas-fork/AMXPs/J1444/snellius_runs/

