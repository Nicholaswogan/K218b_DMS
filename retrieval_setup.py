from picaso import justdoit as jdi
import numpy as np
import os
import utils

def quantile_to_uniform(quantile, lower_bound, upper_bound):
    return quantile*(upper_bound - lower_bound) + lower_bound

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

def model_miri(cube, data): 
    log10CH4, log10CO2, log10C2H6S, log10C2H6S2, T, log10P_ref, log10Ptop_cld, offset_miri = cube

    # PICASO stuff
    opa = PICASO_OPAS
    case1 = PICASO_PLAN

    # Compute spectrum at high res
    wavl_h, rprs2_h = utils.model_spectrum(opa, case1, T, log10CH4, log10CO2, log10C2H6S, log10C2H6S2, log10P_ref, log10Ptop_cld)

    # Rebin the spectrum to miri
    rprs2_model_at_data = utils.fit_model_to_data(data['miri']['wv_bins'].copy(), wavl_h.copy(), rprs2_h.copy())

    # Apply offset
    rprs2_model_at_data += offset_miri

    return rprs2_model_at_data

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

def model_miri_noDMS(cube, data): 
    log10CH4, log10CO2, T, log10P_ref, log10Ptop_cld, offset_miri = cube

    log10C2H6S, log10C2H6S2 = -20.0, -20.0

    # PICASO stuff
    opa = PICASO_OPAS
    case1 = PICASO_PLAN

    # Compute spectrum at high res
    wavl_h, rprs2_h = utils.model_spectrum(opa, case1, T, log10CH4, log10CO2, log10C2H6S, log10C2H6S2, log10P_ref, log10Ptop_cld)

    # Rebin the spectrum to miri
    rprs2_model_at_data = utils.fit_model_to_data(data['miri']['wv_bins'].copy(), wavl_h.copy(), rprs2_h.copy())

    # Apply offset
    rprs2_model_at_data += offset_miri

    return rprs2_model_at_data

def prior_miri_noDMS(cube):

    params = np.empty(len(cube))
    for i in range(len(params)):
        params[i] = quantile_to_uniform(cube[i], *PRIORS_MIRI_NODMS[i])

    return params