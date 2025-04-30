import os
import numpy as np
import pickle
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)
import utils
import retrieval_setup

import warnings
warnings.filterwarnings('ignore')

# Log Likelyhood
def loglike(cube):
    data = DATA_DICT
    y, e = np.zeros((0)), np.zeros((0))
    for key in DATA_KEYS:
        y = np.append(y, data[key]['rprs2'])
        e = np.append(e, data[key]['rprs2_err'])

    within_implicit_priors = IMPLICIT_PRIORS(cube)

    # Compute model spectrum
    if within_implicit_priors:
        resulty = MODEL(cube, data)
        loglikelihood = -0.5*np.sum((y - resulty)**2/e**2)
        return loglikelihood
    else:
        return -1.0e100

RETRIEVAL_INPUTS = { 
    'miri': {
        'data': utils.get_data(),
        'keys': ['miri'],
        'model': retrieval_setup.model_miri,
        'implicit': retrieval_setup.check_implicit_prior_miri,
        'prior': retrieval_setup.prior_miri,
        'params': retrieval_setup.NAMES_MIRI
    },
    'miri_noDMS': {
        'data': utils.get_data(),
        'keys': ['miri'],
        'model': retrieval_setup.model_miri_noDMS,
        'implicit': retrieval_setup.check_implicit_prior_miri_noDMS,
        'prior': retrieval_setup.prior_miri_noDMS,
        'params': retrieval_setup.NAMES_MIRI_NODMS
    },
    'mirip': {
        'data': utils.get_data(),
        'keys': ['miri'],
        'model': retrieval_setup.model_mirip,
        'implicit': retrieval_setup.check_implicit_prior_mirip,
        'prior': retrieval_setup.prior_mirip,
        'params': retrieval_setup.NAMES_MIRIP
    },
    'mirip_noDMS': {
        'data': utils.get_data(),
        'keys': ['miri'],
        'model': retrieval_setup.model_mirip_noDMS,
        'implicit': retrieval_setup.check_implicit_prior_mirip_noDMS,
        'prior': retrieval_setup.prior_mirip_noDMS,
        'params': retrieval_setup.NAMES_MIRIP_NODMS
    },
    'all': {
        'data': utils.get_data(),
        'keys': ['soss','nrs1','nrs2','miri'],
        'model': retrieval_setup.model_all,
        'implicit': retrieval_setup.check_implicit_prior_all,
        'prior': retrieval_setup.prior_all,
        'params': retrieval_setup.NAMES_ALL
    },
    'all_noDMS': {
        'data': utils.get_data(),
        'keys': ['soss','nrs1','nrs2','miri'],
        'model': retrieval_setup.model_all_noDMS,
        'implicit': retrieval_setup.check_implicit_prior_all_noDMS,
        'prior': retrieval_setup.prior_all_noDMS,
        'params': retrieval_setup.NAMES_ALL_NODMS
    },
}

if __name__ == '__main__':

    # mpiexec -n <number of processes> python retrieval_run.py

    sampling = 'ultranest'
    models_to_run = ['mirip','mirip_noDMS']
    for model_name in models_to_run:

        #~~~ Sets stuff here ~~~#
        DATA_DICT = RETRIEVAL_INPUTS[model_name]['data']
        DATA_KEYS = RETRIEVAL_INPUTS[model_name]['keys']
        IMPLICIT_PRIORS = RETRIEVAL_INPUTS[model_name]['implicit']
        MODEL = RETRIEVAL_INPUTS[model_name]['model']
        PRIOR = RETRIEVAL_INPUTS[model_name]['prior']
        PARAM_NAMES = RETRIEVAL_INPUTS[model_name]['params']
        #~~~ End settings stuff ~~~#

        if sampling == 'ultranest':
            import ultranest

            out_dir = f'ultranest/{model_name}'

            # Make sampler
            sampler = ultranest.ReactiveNestedSampler(
                PARAM_NAMES,
                loglike,
                PRIOR,
                log_dir=out_dir, 
                resume='resume'
            )

            # Retrieval
            results = sampler.run(
                min_num_live_points=500,
            )

            pickle.dump(results, open(f'{out_dir}/{model_name}.pkl','wb'))

        elif sampling == 'pymultinest':
            from pymultinest.solve import solve

            outputfiles_basename = f'pymultinest/{model_name}/{model_name}'
            if not os.path.isdir(f'pymultinest/{model_name}'):
                os.mkdir(f'pymultinest/{model_name}')

            results = solve(
                LogLikelihood=loglike, 
                Prior=PRIOR, 
                n_dims=len(PARAM_NAMES), 
                outputfiles_basename=outputfiles_basename, 
                verbose=True
            )

            # Save pickle
            pickle.dump(results, open(outputfiles_basename+'.pkl','wb'))
