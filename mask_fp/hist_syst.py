import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import glob
import fitsio
from astropy.io import fits
import sys

def get_ra_dec(filename):
    data = fitsio.FITS(filename)
    ra = data[1]['RA'].read()
    dec = data[1]['DEC'].read()

    return ra, dec
    
def get_dens(ra,dec,nside, nest=True):
    hp_indices = hp.ang2pix(nside, (90-dec)*np.pi/180, ra*np.pi/180, nest=nest)
    hp_ind_unique, hp_ind_count = np.unique(hp_indices, return_counts=True)
    pixarea = hp.nside2pixarea(nside, degrees=True)
    nb_pix_filled = len(hp_ind_unique)
    nb_pix_tot = hp.nside2npix(nside)
    hp_density = np.zeros(nb_pix_tot)
    hp_density[hp_ind_unique] = hp_ind_count/pixarea

    return hp_density
    
def mask_pos(filename,cols):
	hdul = fits.open(filename)
	data = hdul[1].data
	hdul.close()
	mask = data.field(cols[0])>0.
	for i in range(1,len(cols)):
		mask &= (data.field(cols[i]) > 0.)
	return mask
	
def get_mask(pixfile,col):
	hdul = fits.open(pixfile)
	data = hdul[1].data
	hdul.close()
	mask = data.field(col)
	if 'FRACAREA' in col:
		mask = mask > 0.
	return mask

def plt_hpsyst(filenames_data, filename_pixweight='pixweight-dr10-128-new.fits', cols=[], nside=128, do_mask=False, mask=[], nbins=20, nbmin_hp_perbins=100, nb_rows=2, figsize=(15,5), 
			   ylim=[-0.1,0.1], stardens=0, savename=None, color = ['tab:blue', 'tab:orange'], nest=True, labels=[], weights=None):
    if not isinstance(filenames_data,list):
        filenames_data=[filenames_data]

    if isinstance(filename_pixweight,str):
        hdul = fits.open(filename_pixweight)
        cat_hp = hdul[1].data
        hdul.close()
    else:
        cat_hp = filename_pixweight 

    if len(labels)==0:
        labels = [None]*len(filenames_data)
    if weights is None:
         weights = [np.ones(hp.nside2npix(nside))]*len(filenames_data)
    elif not isinstance(weights, list):
         weights = [weights]
        
    for (kk,filename_data),c,label, weight in zip(enumerate(filenames_data),color,labels, weights):
        
        if isinstance(filename_data,str):
            ra,dec = get_ra_dec(filename_data)
        else:
            ra, dec = filename_data['RA'], filename_data['DEC']
        dens = get_dens(ra,dec,nside,nest) * weight
        if do_mask:
            dens = dens[mask]
        
        nb_plot = len(cols)
        nb_cols = (nb_plot+(nb_rows-1))//nb_rows
        if kk == 0: 
            fig,axes = plt.subplots(nb_rows,nb_cols,figsize=figsize, sharey=True)
            axes=axes.flatten()
    #		c = ['tab:blue', 'tab:orange']

        keep_tot = dens != 0.0
        mean_density = np.mean(dens[keep_tot])

        for i,ax in enumerate(axes):
            #ax = fig.add_subplot(nb_rows, nb_cols, i+1)
            col_name = cols[i]
            data_name = cat_hp[col_name]
            if do_mask:
                data_name = data_name[mask]
            
            minmax = np.quantile(data_name,q=[0.02,0.98])
            min_data = minmax[0]
            max_data = minmax[1]
            
            if 'GALDEPTH' in col_name: 
                data_name = -2.5*(np.log10(5/np.sqrt(data_name))-9)
                min_data = -2.5*(np.log10(5/np.sqrt(min_data))-9)
                max_data = -2.5*(np.log10(5/np.sqrt(max_data))-9)
            if 'STARDENS' in col_name:
                data_name = np.log10(data_name)
                min_data = np.log10(min_data)
                max_data = np.log10(max_data)
            
            sizebin = (max_data - min_data)/(nbins)
            keep_sum = np.zeros(nbins)
            center_bins = np.zeros(nbins)
            nx = np.zeros(nbins)
            nx_std = np.zeros(nbins)
            
            for j in range(nbins):
            
                keep = np.all([keep_tot,data_name>=min_data+sizebin*j, data_name<=min_data+sizebin*(j+1)], axis=0)
                #(col_name, '::  Nb hp for the x-bin (', min_data+sizebin*j, '<x<', min_data+sizebin*(j+1),'): ', np.sum(keep), ' ( nb hp min=', nbmin_hp_perbins, ') ')
                keep_sum[j] = np.sum(keep)
                nx[j] = np.mean(dens[keep]/mean_density)
                nx_std[j] = np.std(dens[keep]/mean_density)
                center_bins[j] = min_data+sizebin*(j+0.5)
            
            ax.set_xlabel(col_name)
            
            keep_plt = keep_sum >= nbmin_hp_perbins
            ax.errorbar(center_bins[keep_plt], nx[keep_plt] - 1, yerr=nx_std[keep_plt]/np.sqrt(keep_sum[keep_plt]), label=label, color=c)
            ax.axhline(0,ls=':',c='k')
            
            hist_count, bin_edges = np.histogram(data_name[keep_tot], bins=nbins, range=(min_data, max_data), weights=np.ones(np.sum(keep_tot))/np.sum(keep_tot)*0.5, density=False)
            x_hist, y_hist = np.ones(2*nbins), np.ones(2*nbins)
            x_hist[::2], x_hist[1::2] = bin_edges[:-1], bin_edges[1:]
            y_hist[::2], y_hist[1::2] = hist_count, hist_count
            shift = ylim[0]
            ax.plot(x_hist, y_hist+shift, color=c)
            ax.fill_between(x_hist, y_hist+shift, y2=shift, alpha=0., color=c)
            ax.set_xlim([center_bins[keep_plt][0]-0.5*sizebin, center_bins[keep_plt][-1]+0.5*sizebin])
            ax.set_ylim(ylim)
    axes[0].set_ylabel(r"$\frac{n - \overline{n}}{\overline{n}}$",fontsize=18) 
    axes[4].set_ylabel(r"$\frac{n - \overline{n}}{\overline{n}}$",fontsize=18) 
    if label is not None:
        ax.legend()
    fig.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    #plt.close()

##########################################################################################################################################################################################
##########################################################################################################################################################################################
if __name__ == '__main__':
	pix = 'pixweight-dr10-128-new.fits'
	footprint = 'Legacy_Imaging_DR10_footprint_128.fits'
	cat_lrg = '/home/averdier/Documents/EPFL/DOCTORAT/4MOST/MAY2024/data_2/target_lrg_4most_v2.fits'
	cat_bg = '/home/averdier/Documents/EPFL/DOCTORAT/4MOST/MAY2024/data_2/target_bg_mag_r19_4most_v2.fits'
	columns = ['EBV','GALDEPTH_G','GALDEPTH_R','GALDEPTH_Z','PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','STARDENS']
	#mask_t = mask_pos(pix,columns)
	mask_t = get_mask(footprint,'IS4MOST_LRG') & get_mask(footprint,'ISSGC') & get_mask(pix,'FRACAREA_12290')
	savename = 'test_hist'
	plt_hpsyst([cat_lrg,cat_bg],cols=columns,do_mask = True,mask=mask_t,savename=savename,labels=['LRG','BG'])
