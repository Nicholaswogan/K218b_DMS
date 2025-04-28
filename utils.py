
import numpy as np
from photochem.utils import stars
import pickle
import pandas as pd

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

def get_data():
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