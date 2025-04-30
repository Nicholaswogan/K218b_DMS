from picaso import justdoit as jdi
import numpy as np
import os
import utils
from scipy import stats

def quantile_to_uniform(quantile, lower_bound, upper_bound):
    return quantile*(upper_bound - lower_bound) + lower_bound

def quantile_to_truncgauss(quantile, low, high, mu, sigma):
    return stats.truncnorm.ppf(quantile, low/sigma - mu/sigma, high/sigma - mu/sigma, loc=mu, scale=sigma)

def create_picaso():
    filename_db = os.path.join('lupu_0.5_15_R10000.db')
    opa = jdi.opannection(filename_db=filename_db)

    case1 = jdi.inputs()

    K218b = utils.K218b

    case1.gravity(
        mass=K218b.planet_mass, 
        mass_unit=jdi.u.Unit('M_earth'),
        radius=K218b.planet_radius, 
        radius_unit=jdi.u.Unit('R_earth')
    )

    case1.star(
        opa, 
        temp=K218b.stellar_Teff,
        metal=K218b.stellar_metal,
        logg=K218b.stellar_logg,
        radius=K218b.stellar_radius, 
        radius_unit=jdi.u.Unit('R_sun'),
        database='phoenix'
    )
    return opa, case1

# Global variables
PICASO_OPAS, PICASO_PLAN = create_picaso()

##############
#~~~ miri ~~~#
##############

PARAM_MIRI = [
    ['log10CH4', [-13.0, -0.3]], 
    ['log10CO2', [-13.0, -0.3]], 
    ['log10C2H6S', [-13.0, -0.3]], 
    ['log10C2H6S2', [-13.0, -0.3]], 
    ['T', [100.0, 500.0]],
    ['log10P_ref', [-6.0, 0.0]],
    ['log10Ptop_cld', [-6.0, 1.0]],
    ['offset_miri', [-100.0e-6, 100.0e-6]]
]
NAMES_MIRI = [a[0] for a in PARAM_MIRI]
PRIORS_MIRI = [a[1] for a in PARAM_MIRI]

def model_miri(cube, data, wv_bins=None): 
    log10CH4, log10CO2, log10C2H6S, log10C2H6S2, T, log10P_ref, log10Ptop_cld, offset_miri = cube

    # PICASO stuff
    opa = PICASO_OPAS
    case1 = PICASO_PLAN

    # Compute spectrum at high res
    wavl_h, rprs2_h = utils.model_spectrum(opa, case1, T, log10CH4, log10CO2, log10C2H6S, log10C2H6S2, log10P_ref, log10Ptop_cld)

    if wv_bins is not None:
        return utils.fit_model_to_data(wv_bins.copy(), wavl_h.copy(), rprs2_h.copy())
    
    # Rebin the spectrum to miri
    rprs2_model_at_data = utils.fit_model_to_data(data['miri']['wv_bins'].copy(), wavl_h.copy(), rprs2_h.copy())

    # Apply offset
    rprs2_model_at_data += offset_miri

    return rprs2_model_at_data

def check_implicit_prior_miri(cube):
    log10CH4, log10CO2, log10C2H6S, log10C2H6S2, T, log10P_ref, log10Ptop_cld, offset_miri = cube
    within_implicit_priors = True
    if np.sum(10.0**np.array([log10CH4, log10CO2, log10C2H6S, log10C2H6S2])) > 1.0:
        within_implicit_priors = False
    return within_implicit_priors 

def prior_miri(cube):

    params = np.empty(len(cube))
    for i in range(len(params)):
        params[i] = quantile_to_uniform(cube[i], *PRIORS_MIRI[i])

    return params

####################
#~~~ miri_noDMS ~~~#
####################

PARAM_MIRI_NODMS = [
    ['log10CH4', [-13.0, -0.3]], 
    ['log10CO2', [-13.0, -0.3]], 
    ['T', [100.0, 500.0]],
    ['log10P_ref', [-6.0, 0.0]],
    ['log10Ptop_cld', [-6.0, 1.0]],
    ['offset_miri', [-100.0e-6, 100.0e-6]]
]
NAMES_MIRI_NODMS = [a[0] for a in PARAM_MIRI_NODMS]
PRIORS_MIRI_NODMS = [a[1] for a in PARAM_MIRI_NODMS]

def model_miri_noDMS(cube, data, wv_bins=None): 
    log10CH4, log10CO2, T, log10P_ref, log10Ptop_cld, offset_miri = cube

    log10C2H6S, log10C2H6S2 = -20.0, -20.0

    # PICASO stuff
    opa = PICASO_OPAS
    case1 = PICASO_PLAN

    # Compute spectrum at high res
    wavl_h, rprs2_h = utils.model_spectrum(opa, case1, T, log10CH4, log10CO2, log10C2H6S, log10C2H6S2, log10P_ref, log10Ptop_cld)

    if wv_bins is not None:
        return utils.fit_model_to_data(wv_bins.copy(), wavl_h.copy(), rprs2_h.copy())

    # Rebin the spectrum to miri
    rprs2_model_at_data = utils.fit_model_to_data(data['miri']['wv_bins'].copy(), wavl_h.copy(), rprs2_h.copy())

    # Apply offset
    rprs2_model_at_data += offset_miri

    return rprs2_model_at_data

def check_implicit_prior_miri_noDMS(cube):
    log10CH4, log10CO2, T, log10P_ref, log10Ptop_cld, offset_miri = cube
    within_implicit_priors = True
    if np.sum(10.0**np.array([log10CH4, log10CO2])) > 1.0:
        within_implicit_priors = False
    return within_implicit_priors 

def prior_miri_noDMS(cube):

    params = np.empty(len(cube))
    for i in range(len(params)):
        params[i] = quantile_to_uniform(cube[i], *PRIORS_MIRI_NODMS[i])

    return params

#############
#~~~ all ~~~#
#############

PARAM_ALL = [
    ['log10CH4', [-13.0, -0.3]], 
    ['log10CO2', [-13.0, -0.3]], 
    ['log10C2H6S', [-13.0, -0.3]], 
    ['log10C2H6S2', [-13.0, -0.3]], 
    ['T', [100.0, 500.0]],
    ['log10P_ref', [-6.0, 0.0]],
    ['log10Ptop_cld', [-6.0, 1.0]],
    ['offset_miri', [-100.0e-6, 100.0e-6]],
    ['offset_soss', [-1000.0e-6, 1000.0e-6]],
    ['offset_nrs1', [-1000.0e-6, 1000.0e-6]],
    ['offset_nrs2', [-1000.0e-6, 1000.0e-6]]
]
NAMES_ALL = [a[0] for a in PARAM_ALL]
PRIORS_ALL = [a[1] for a in PARAM_ALL]

def model_all(cube, data, wv_bins=None): 
    log10CH4, log10CO2, log10C2H6S, log10C2H6S2, T, log10P_ref, log10Ptop_cld, offset_miri, offset_soss, offset_nrs1, offset_nrs2 = cube

    # PICASO stuff
    opa = PICASO_OPAS
    case1 = PICASO_PLAN

    # Compute spectrum at high res
    wavl_h, rprs2_h = utils.model_spectrum(opa, case1, T, log10CH4, log10CO2, log10C2H6S, log10C2H6S2, log10P_ref, log10Ptop_cld)

    if wv_bins is not None:
        return utils.fit_model_to_data(wv_bins.copy(), wavl_h.copy(), rprs2_h.copy())

    # Rebin the spectrum
    rprs2_model_at_soss = utils.fit_model_to_data(data['soss']['wv_bins'].copy(), wavl_h.copy(), rprs2_h.copy())
    rprs2_model_at_soss += offset_soss

    rprs2_model_at_nrs1 = utils.fit_model_to_data(data['nrs1']['wv_bins'].copy(), wavl_h.copy(), rprs2_h.copy())
    rprs2_model_at_nrs1 += offset_nrs1

    rprs2_model_at_nrs2 = utils.fit_model_to_data(data['nrs2']['wv_bins'].copy(), wavl_h.copy(), rprs2_h.copy())
    rprs2_model_at_nrs2 += offset_nrs2

    rprs2_model_at_miri = utils.fit_model_to_data(data['miri']['wv_bins'].copy(), wavl_h.copy(), rprs2_h.copy())
    rprs2_model_at_miri += offset_miri

    rprs2_model_at_data = np.concatenate((rprs2_model_at_soss, rprs2_model_at_nrs1, rprs2_model_at_nrs2, rprs2_model_at_miri))

    return rprs2_model_at_data

def check_implicit_prior_all(cube):
    log10CH4, log10CO2, log10C2H6S, log10C2H6S2, T, log10P_ref, log10Ptop_cld, offset_miri, offset_soss, offset_nrs1, offset_nrs2 = cube
    within_implicit_priors = True
    if np.sum(10.0**np.array([log10CH4, log10CO2, log10C2H6S, log10C2H6S2])) > 1.0:
        within_implicit_priors = False
    return within_implicit_priors 

def prior_all(cube):

    params = np.empty(len(cube))
    for i in range(len(params)):
        params[i] = quantile_to_uniform(cube[i], *PRIORS_ALL[i])

    return params

###################
#~~~ all_noDMS ~~~#
###################

PARAM_ALL_NODMS = [
    ['log10CH4', [-13.0, -0.3]], 
    ['log10CO2', [-13.0, -0.3]], 
    ['T', [100.0, 500.0]],
    ['log10P_ref', [-6.0, 0.0]],
    ['log10Ptop_cld', [-6.0, 1.0]],
    ['offset_miri', [-100.0e-6, 100.0e-6]],
    ['offset_soss', [-1000.0e-6, 1000.0e-6]],
    ['offset_nrs1', [-1000.0e-6, 1000.0e-6]],
    ['offset_nrs2', [-1000.0e-6, 1000.0e-6]]
]
NAMES_ALL_NODMS = [a[0] for a in PARAM_ALL_NODMS]
PRIORS_ALL_NODMS = [a[1] for a in PARAM_ALL_NODMS]

def model_all_noDMS(cube, data, wv_bins=None): 
    log10CH4, log10CO2, T, log10P_ref, log10Ptop_cld, offset_miri, offset_soss, offset_nrs1, offset_nrs2 = cube

    log10C2H6S = -20.0
    log10C2H6S2 = -20.0

    # PICASO stuff
    opa = PICASO_OPAS
    case1 = PICASO_PLAN

    # Compute spectrum at high res
    wavl_h, rprs2_h = utils.model_spectrum(opa, case1, T, log10CH4, log10CO2, log10C2H6S, log10C2H6S2, log10P_ref, log10Ptop_cld)

    if wv_bins is not None:
        return utils.fit_model_to_data(wv_bins.copy(), wavl_h.copy(), rprs2_h.copy())

    # Rebin the spectrum
    rprs2_model_at_soss = utils.fit_model_to_data(data['soss']['wv_bins'].copy(), wavl_h.copy(), rprs2_h.copy())
    rprs2_model_at_soss += offset_soss

    rprs2_model_at_nrs1 = utils.fit_model_to_data(data['nrs1']['wv_bins'].copy(), wavl_h.copy(), rprs2_h.copy())
    rprs2_model_at_nrs1 += offset_nrs1

    rprs2_model_at_nrs2 = utils.fit_model_to_data(data['nrs2']['wv_bins'].copy(), wavl_h.copy(), rprs2_h.copy())
    rprs2_model_at_nrs2 += offset_nrs2

    rprs2_model_at_miri = utils.fit_model_to_data(data['miri']['wv_bins'].copy(), wavl_h.copy(), rprs2_h.copy())
    rprs2_model_at_miri += offset_miri

    rprs2_model_at_data = np.concatenate((rprs2_model_at_soss, rprs2_model_at_nrs1, rprs2_model_at_nrs2, rprs2_model_at_miri))

    return rprs2_model_at_data

def check_implicit_prior_all_noDMS(cube):
    log10CH4, log10CO2, T, log10P_ref, log10Ptop_cld, offset_miri, offset_soss, offset_nrs1, offset_nrs2 = cube
    within_implicit_priors = True
    if np.sum(10.0**np.array([log10CH4, log10CO2])) > 1.0:
        within_implicit_priors = False
    return within_implicit_priors 

def prior_all_noDMS(cube):

    params = np.empty(len(cube))
    for i in range(len(params)):
        params[i] = quantile_to_uniform(cube[i], *PRIORS_ALL_NODMS[i])

    return params
