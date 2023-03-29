import dill as pickle

numenergies=32
n_params=4

with open(f'run_A{n_params}/LikelihoodDiagnostics_numenergies={numenergies}.pkl', 'rb') as file:
     ldict = pickle.load(file)
     ldict.popitem()
     
for key in ldict:
    if isinstance(ldict[key], dict):
        print('numenergies: ', numenergies)
        print('deltatime: ', ldict[key]['deltatime'])
        print('p: ', ldict[key]['p'])
        print('loglikelihood: ', ldict[key]['loglikelihood'])
        print('\n')
        
