import dill as pickle

numenergies=32
n_params=4
xpsi_size=4

ldict = {}
for rank in range(xpsi_size): 
    with open(f'run_A{n_params}/LikelihoodDiagnostics_ne={numenergies}_rank={rank}.pkl', 'rb') as file:
    #with open(f'helios_runs/run_A4/304151/run_A4/LikelihoodDiagnostics_ne={numenergies}_rank={rank}.pkl', 'rb') as file:
    #with open(f'helios_runs/run_A5/304152/run_A5/LikelihoodDiagnostics_numenergies={numenergies}.pkl', 'rb') as file:
         ldict[rank] = pickle.load(file)
         ldict[rank].popitem()
     
for key in ldict:
    if isinstance(ldict[key], dict):
        print('numenergies: ', numenergies)
        print('deltatime: ', ldict[key]['deltatime'])
        print('p: ', ldict[key]['p'])
        print('loglikelihood: ', ldict[key]['loglikelihood'])
        print('\n')
        
