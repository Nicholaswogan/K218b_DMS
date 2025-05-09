
import numpy as np
from photochem.utils import stars
import pickle
import pandas as pd
from scipy import optimize
from scipy import special
import warnings

class K218b:
    planet_radius = 2.610 # Earth radii
    planet_mass = 8.63 # Earth masses
    planet_Teq = 278.7 # Equilibrium temp (K)
    stellar_radius = 0.4445 # Solar radii
    stellar_Teff = 3457 # K
    stellar_metal = 0.12 # log10(metallicity)
    stellar_logg = 4.79 # log10(gravity), in cgs units

def miri_data():

    wv, wv_err, rprs2, rprs2_err = np.loadtxt('data/K2-18b_miri_lrs_jexores.txt',skiprows=1).T
    
    wv_bins = np.empty((len(wv),2))
    for i in range(len(wv)):
        wv_bins[i,0] = wv[i] - wv_err[i]
        wv_bins[i,1] = wv[i] + wv_err[i]

    out = {
        'wv': wv,
        'rprs2': rprs2,
        'rprs2_err': rprs2_err,
        'wv_err': wv_err,
        'wv_bins': wv_bins
    }
    return out

def get_data_old():
    # Get NIRSpec and NIRISS
    with open('data/lowres.pkl','rb') as f:
        data = pickle.load(f)
    del data['all']

    # Add miri data
    miri = miri_data()
    data['miri'] = miri

    # Split G395H into NRS1 and NRS2
    nrs1 = {}
    nrs2 = {}
    inds1 = np.where(data['g395h']['wv'] < 3.78)
    inds2 = np.where(data['g395h']['wv'] >= 3.78)
    for key in data['g395h']:
        # if key == 'wv_bins':
        nrs1[key] = data['g395h'][key][inds1]
        nrs2[key] = data['g395h'][key][inds2]
    del data['g395h']
    data['nrs1'] = nrs1
    data['nrs2'] = nrs2

    # Compute all data
    out = {}
    for key in data['miri']:
        out[key] = np.zeros((0))
    out['wv_bins'] = np.zeros((0,2))
    for name in ['soss','nrs1','nrs2','miri']:
        for key in out:
            out[key] = np.concatenate((out[key],data[name][key]))
    data['all'] = out

    return data

def get_madhu_data(filename):

    wv, wv_err, rprs2, rprs2_err = np.loadtxt(filename,skiprows=1).T
    
    wv_bins = np.empty((len(wv),2))
    for i in range(len(wv)):
        wv_bins[i,0] = wv[i] - wv_err[i]
        wv_bins[i,1] = wv[i] + wv_err[i]

    out = {
        'wv': wv,
        'rprs2': rprs2,
        'rprs2_err': rprs2_err,
        'wv_err': wv_err,
        'wv_bins': wv_bins
    }
    return out

def split_g395h(g395h):
    # Split G395H into NRS1 and NRS2
    nrs1 = {}
    nrs2 = {}
    inds1 = np.where(g395h['wv'] < 3.78)
    inds2 = np.where(g395h['wv'] >= 3.78)
    for key in g395h:
        nrs1[key] = g395h[key][inds1]
        nrs2[key] = g395h[key][inds2]
    return nrs1, nrs2

def get_data(lowres=True):

    if lowres:
        soss = get_madhu_data('data/K2-18b_niriss_soss_lowres.txt')
        g395h = get_madhu_data('data/K2-18b_nirspec_g395h_lowres.txt')
    else:
        soss = get_madhu_data('data/K2-18b_niriss_soss_native.txt')
        g395h = get_madhu_data('data/K2-18b_nirspec_g395h_native.txt')
    nrs1, nrs2 = split_g395h(g395h)
    miri = get_madhu_data('data/K2-18b_miri_lrs_jexores.txt')

    # Collect all data
    data = {
        'soss': soss,
        'nrs1': nrs1,
        'nrs2': nrs2,
        'miri': miri,
    }
    
    # Compute all data
    out = {}
    for key in data['miri']:
        out[key] = np.zeros((0))
    out['wv_bins'] = np.zeros((0,2))
    for name in ['soss','nrs1','nrs2','miri']:
        for key in out:
            out[key] = np.concatenate((out[key],data[name][key]))
    data['all'] = out

    return data

def spectrum(opa, case1, atm, log10Ptop_cld=None, atmosphere_kwargs={}):

    # Set atmosphere
    case1.atmosphere(atm, verbose=False, **atmosphere_kwargs)

    if log10Ptop_cld is not None:
        ptop = log10Ptop_cld
        pbottom = 2 #at depth
        dp =  pbottom - ptop
        case1.clouds(p=[2],dp=[dp],opd=[10], g0=[0],w0=[0.99])
    else:
        # No clouds
        case1.clouds_reset()
        
    # Compute spectrum 
    df = case1.spectrum(opa, calculation='transmission')

    # Extract spectrum
    wv_h = 1e4/df['wavenumber'][::-1].copy()
    wavl_h = stars.make_bins(wv_h)
    rprs2_h = df['transit_depth'][::-1].copy()

    return wavl_h, rprs2_h

def fit_model_to_data(wv_bins_data, wavl_model, rprs2_model):
    
    rprs2_model_at_data = np.empty(wv_bins_data.shape[0])
    for i in range(wv_bins_data.shape[0]):
        res = stars.rebin(wavl_model, rprs2_model, wv_bins_data[i,:].copy())
        rprs2_model_at_data[i] = res[0]

    return rprs2_model_at_data

def regrid_model(wavl_h, rprs2_h, R):
    wavl = stars.grid_at_resolution(np.min(wavl_h), np.max(wavl_h), R)
    wv = (wavl[1:] + wavl[:-1])/2
    rprs2 = stars.rebin(wavl_h.copy(), rprs2_h.copy(), wavl.copy())
    return wv, wavl, rprs2

def wv_bins_at_resolution(min_wv, max_wv, R):
    wavl = stars.grid_at_resolution(min_wv, max_wv, R)
    wv = (wavl[1:] + wavl[:-1])/2
    wv_bins = np.empty((len(wavl)-1,2))
    for i in range(len(wavl)-1):
        wv_bins[i,0] = wavl[i]
        wv_bins[i,1] = wavl[i+1]
    return wv, wavl, wv_bins

def model_spectrum(opa, case1, T, log10CH4, log10CO2, log10C2H6S, log10C2H6S2, log10P_ref, log10Ptop_cld, nz=60, atmosphere_kwargs={}):

    f_CH4, f_CO2, f_C2H6S, f_C2H6S2 = 10.0**log10CH4, 10.0**log10CO2, 10.0**log10C2H6S, 10.0**log10C2H6S2
    
    f_H2 = 1.0 - (f_CH4 + f_CO2 + f_C2H6S + f_C2H6S2)
    assert f_H2 > 0.0

    case1.inputs['approx']['p_reference'] = 10.0**log10P_ref

    # Pressure
    P = np.logspace(-7,2,nz)
    
    # Create dict
    atm = {
        'pressure': P,
        'temperature': np.ones(nz)*T,
        'CH4': np.ones(nz)*f_CH4,
        'CO2': np.ones(nz)*f_CO2,
        'C2H6S': np.ones(nz)*f_C2H6S,
        'C2H6S2': np.ones(nz)*f_C2H6S2,
        'H2': np.ones(nz)*f_H2,
    }
    atm = pd.DataFrame(atm)

    wavl_h, rprs2_h = spectrum(opa, case1, atm, log10Ptop_cld, atmosphere_kwargs)

    return wavl_h, rprs2_h

def rho_fcn(sigma):
    return 1 - special.erf(sigma/np.sqrt(2))

def bayes_factor(sigma):
    r = rho_fcn(sigma)
    return - 1/(np.exp(1)*r*np.log(r))

def objective(x, bayes_factor_input):
    sigma = x[0]
    return bayes_factor(sigma) - bayes_factor_input

def sigma_significance(bayes_factor_input):
    if bayes_factor_input > 1e8:
        warnings.warn("Bayes factors larger than 1e8 can not be computed. Returning sigma = 6.392455915996625")
        return 6.392455915996625
    if bayes_factor_input <= 1:
        return 0.9004526284839545
    initial_cond = np.array([6.0])
    sol = optimize.root(objective, initial_cond, args = (bayes_factor_input,))
    if not sol.success:
        raise Exception("Root solving failed: "+sol.message)
    return sol.x[0]

def detection_sigma(lnB):
    """Computes detection sigma from bayes factor.

    Parameters
    ----------
    lnB : float
        The natural log of the bayes factor

    Returns
    -------
    float
        Detection "sigma" significance.

    """
    if lnB < np.log(2e1):
        return sigma_significance(np.exp(lnB))
    
    logp = np.arange(-100.00,-0.00,.01) #reverse order
    logp = logp[::-1] # original order
    P = 10.0**logp
    Barr = -1./(np.exp(1)*P*np.log(P))

    sigma = np.arange(0.1,100.10,.01)
    p_p = special.erfc(sigma/np.sqrt(2.0))

    B = np.exp(lnB)
    pvalue = 10.0**np.interp(np.log10(B),np.log10(Barr),np.log10(P))
    sig = np.interp(pvalue,p_p[::-1],sigma[::-1])

    return sig