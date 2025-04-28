import pickle
import numpy as np
from matplotlib import pyplot as plt
from ultranest import plot as uplot
import retrieval_setup
import matplotlib
from copy import deepcopy
import utils

import warnings
warnings.filterwarnings('ignore')

DATA = utils.get_data()
PICASO_OPAS = retrieval_setup.PICASO_OPAS
PICASO_PLAN = retrieval_setup.PICASO_PLAN

def max_like(result):

    wv, wavl, wv_bins = utils.wv_bins_at_resolution(0.5, 13, 70)

    log10CH4, log10CO2,log10C2H6S, log10C2H6S2, T, log10P_ref, log10Ptop_cld, offset_miri = result['maximum_likelihood']['point'][:8]
    
    all_species = ['CH4','CO2','C2H6S','C2H6S2']
    cases = {'all':[],'cloud': all_species}
    for sp in all_species:
        tmp = deepcopy(all_species)
        tmp.remove(sp)
        cases[sp] = tmp
    
    res = {}
    for case in cases:
        log10Ptop_cld_copy = deepcopy(log10Ptop_cld)
        if case in all_species:
            log10Ptop_cld_copy = 1.99
        wavl_h, rprs2_h = utils.model_spectrum(PICASO_OPAS, PICASO_PLAN, T, log10CH4, log10CO2, log10C2H6S, log10C2H6S2, log10P_ref, log10Ptop_cld_copy,
                                              atmosphere_kwargs={'exclude_mol': cases[case]})
        rprs2 = utils.fit_model_to_data(wv_bins.copy(), wavl_h.copy(), rprs2_h.copy())
        rprs2 += offset_miri
        res[case] = {'wv': wv, 'rprs2': rprs2}

    return res

def make_band(model, result, ind_miri, wv, wv_bins, n):

    inds = np.random.randint(0,result['samples'].shape[0]-1,n)
    samples = result['samples'][inds,:]
    
    band = uplot.PredictionBand(wv)
    
    for i,x in enumerate(samples):
        print(i,end='\r')
        rprs2 = model(x, DATA, wv_bins)
        rprs2 += x[ind_miri]
        rprs2 *= 1e2
        band.add(rprs2)

    return band

def main():

    data = DATA

    # Read in results
    key = 'miri'
    with open('ultranest/'+key+'/'+key+'.pkl','rb') as f:
        result_miri = pickle.load(f)

    key = 'miri_noDMS'
    with open('ultranest/'+key+'/'+key+'.pkl','rb') as f:
        result_miri_noDMS = pickle.load(f)

    key = 'all'
    with open('ultranest/'+key+'/'+key+'.pkl','rb') as f:
        result_all = pickle.load(f)

    key = 'all_noDMS'
    with open('ultranest/'+key+'/'+key+'.pkl','rb') as f:
        result_all_noDMS = pickle.load(f)

    # Get max likelihood
    res_miri = max_like(result_miri)
    res_all = max_like(result_all)

    # get band_miri and band_all
    np.random.seed(0)
    wv, wavl, wv_bins = utils.wv_bins_at_resolution(0.5, 13, 70)
    band_miri = make_band(retrieval_setup.model_miri, result_miri, -1, wv, wv_bins, 200)
    band_all = make_band(retrieval_setup.model_all, result_all, -4, wv, wv_bins, 200)

    # with open('bands.pkl','rb') as f:
    #     tmp = pickle.load(f)
    # band_miri = tmp['miri']
    # band_all = tmp['all']

    # Plot
    plt.rcParams.update({'font.size': 13})
    fig = plt.figure(constrained_layout=False,figsize=[11,12])
    fig.patch.set_facecolor("w")

    x = 41
    gs = fig.add_gridspec(200, 100)
    ax1 = fig.add_subplot(gs[:x, :])
    ax2 = fig.add_subplot(gs[x+2:2*x+2, :])
    x1 = 100+(100-(2*x+2))
    ax3 = fig.add_subplot(gs[x1:x1+x, :])
    ax4 = fig.add_subplot(gs[x1+x+2:200, :])
    axs = [ax1,ax2,ax3,ax4]

    ###
    ### Fit to MIRI only
    ###

    ax = axs[0]

    band_miri.line(ax=ax, color='k', label='Retrieval median fit')
    band_miri.shade(ax=ax, color='k', alpha=0.3)
    band_miri.shade(ax=ax, q=0.475, color='k', alpha=0.3)

    a,b = np.loadtxt('data/madu_model_figure2.csv',delimiter=',').T
    ax.plot(a,b,c='blue',label='Madhusudhan+2025\nmedian fit to MIRI')

    result = result_all
    offsets = [
        -(np.median(result['samples'][:,-3])-np.median(result['samples'][:,-4])),
        -(np.median(result['samples'][:,-2])-np.median(result['samples'][:,-4])),
        -(np.median(result['samples'][:,-1])-np.median(result['samples'][:,-4])),
        0
    ]

    labels = ['NIRISS SOSS','NIRSpec G395H','','MIRI LRS']
    colors = ['C9','C1','C1','C3']
    for i,dkey in enumerate(['soss','nrs1','nrs2','miri']):
        ax.errorbar(
            data[dkey]['wv'],(data[dkey]['rprs2']+offsets[i])*1e2,yerr=data[dkey]['rprs2_err']*1e2,xerr=data[dkey]['wv_err'],
            ls='',marker='o', ms=2,c=colors[i],elinewidth=1,capsize=1, capthick=1,alpha=0.7,label=labels[i]
        )

    ax.legend(ncol=3,bbox_to_anchor=(0.0, 1.01), loc='upper left',fontsize=11,frameon=False)

    B = np.exp(result_miri['logz']-result_miri_noDMS['logz'])
    sig = utils.detection_sigma(np.log(B))
    note = r'$\bf{Retrieval}$ $\bf{with}$ $\bf{only}$ $\bf{MIRI}$ $\bf{data}$'+\
    '\nDMS/DMDS det. Bayes Factor = %.1f'%(B)+\
    '\nDMS/DMDS det. significance = %.1f$\sigma$'%(sig)
    ax.text(0.5, 1.02, note, size = 12, ha='center', va='bottom',transform=ax.transAxes)

    rec = matplotlib.patches.Rectangle((-0.125,-1.38), 1.15, 2.73, fill=False, lw=1.5, clip_on=False,transform=ax.transAxes)
    rec = ax.add_patch(rec)
    ax.text(-0.118, 1.3, '(a)', \
                    size = 25, ha='left', va='top',transform=ax.transAxes)

    ax = axs[1]

    res = res_miri
    shift = 0
    ax.plot(res['all']['wv'], (res['all']['rprs2']+shift)*1e2, c='k', lw=1.5, label='Max likelihood')
    ax.plot(res['all']['wv'], (res['cloud']['rprs2']+shift)*1e2, c='brown', ls='--', label='cloud')
    ax.plot(res['all']['wv'], (res['CH4']['rprs2']+shift)*1e2, c='C1', ls='--', label='CH$_4$')
    ax.plot(res['all']['wv'], (res['CO2']['rprs2']+shift)*1e2, c='C2', ls='--', label='CO$_2$')
    ax.plot(res['all']['wv'], (res['C2H6S']['rprs2']+shift)*1e2, c='C4', ls='--', label='DMS')
    ax.plot(res['all']['wv'], (res['C2H6S2']['rprs2']+shift)*1e2, c='cyan', ls='--', label='DMDS')

    ax.legend(ncol=3,bbox_to_anchor=(0.0, 1.01), loc='upper left',fontsize=11,frameon=False)

    ###
    ### Fit to all data
    ###

    ax = axs[2]

    band_all.line(ax=ax, color='k', label='Retrieval median fit')
    band_all.shade(ax=ax, color='k', alpha=0.3)
    band_all.shade(ax=ax, q=0.475, color='k', alpha=0.3)

    a, b = np.loadtxt('data/madu_model_figure2.csv',delimiter=',').T
    ax.plot(a,b,c='blue',label='Madhusudhan+2025\nmedian fit to MIRI')

    labels = ['NIRISS SOSS','NIRSpec G395H','','MIRI LRS']
    for i,dkey in enumerate(['soss','nrs1','nrs2','miri']):
        ax.errorbar(
            data[dkey]['wv'],(data[dkey]['rprs2']+offsets[i])*1e2,yerr=data[dkey]['rprs2_err']*1e2,xerr=data[dkey]['wv_err'],
            ls='',marker='o', ms=2,c=colors[i],elinewidth=1,capsize=1, capthick=1,alpha=0.7,label=labels[i]
        )

    ax.legend(ncol=3,bbox_to_anchor=(0.0, 1.01), loc='upper left',fontsize=11,frameon=False)

    # grid
    B = np.exp(result_all['logz']-result_all_noDMS['logz'])
    sig = utils.detection_sigma(np.log(B))
    note = r'$\bf{Retrieval}$ $\bf{with}$ $\bf{all}$ $\bf{data}$'+\
    '\nDMS/DMDS det. Bayes Factor = %.1f'%(B)+\
    '\nDMS/DMDS det. significance = 0$\sigma$ (no detection)'
    ax.text(0.5, 1.02, note, size = 12, ha='center', va='bottom',transform=ax.transAxes)

    rec = matplotlib.patches.Rectangle((-0.125,-1.38), 1.15, 2.73, fill=False, lw=1.5, clip_on=False,transform=ax.transAxes)
    rec = ax.add_patch(rec)
    ax.text(-0.118, 1.3, '(b)', \
                    size = 25, ha='left', va='top',transform=ax.transAxes)
    
    ax = axs[3]

    res = res_all
    shift = 0
    ax.plot(res['all']['wv'], (res['all']['rprs2']+shift)*1e2, c='k', lw=1.5, label='Max likelihood')
    ax.plot(res['all']['wv'], (res['cloud']['rprs2']+shift)*1e2, c='brown', ls='--', label='cloud')
    ax.plot(res['all']['wv'], (res['CH4']['rprs2']+shift)*1e2, c='C1', ls='--', label='CH$_4$')
    ax.plot(res['all']['wv'], (res['CO2']['rprs2']+shift)*1e2, c='C2', ls='--', label='CO$_2$')
    ax.plot(res['all']['wv'], (res['C2H6S']['rprs2']+shift)*1e2, c='C4', ls='--', label='DMS')
    ax.plot(res['all']['wv'], (res['C2H6S2']['rprs2']+shift)*1e2, c='cyan', ls='--', label='DMDS')

    ax.legend(ncol=3,bbox_to_anchor=(0.0, 1.01), loc='upper left',fontsize=11,frameon=False)

    for ax in axs:
        ax.set_ylim(0.24,0.34)
        ax.set_yticks(np.arange(0.26,0.34,0.02))
        ax.set_xlim(0.6,12.1)
        ax.set_xticks(np.arange(1,13,1))
        ax.set_ylabel('Transit Depth (%)')

    for ax in [axs[1],axs[3]]:
        ax.set_xlabel('Wavelength (microns)')

    for ax in [axs[0],axs[2]]:
        ax.set_xticklabels([])

    plt.savefig('figures/DMS_retrieval.pdf',bbox_inches='tight')

if __name__ == '__main__':
    main()