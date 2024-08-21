#!/usr/bin/env python
# coding: utf-8

import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
from matplotlib.path import Path
import os
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
from astropy.table import Table
import glob
# to avoid this warning:
# WARNING: AstropyDeprecationWarning: Transforming a frame instance to a frame class (as opposed to another frame instance)
# will not be supported in the future.  Either explicitly instantiate the target frame, or first convert the source frame instance
# to a `astropy.coordinates.SkyCoord` and use its `transform_to()` method. [astropy.coordinates.baseframe]
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
import fitsio


# 4most bg & lrg, re-tuned cuts (27Feb2020)
def get_4most_bg_old_vista_sel(j,k,w1,r, jmin = 16, jmax= 18.25, rmax= 22):
    jw1 = j-w1
    jk = j-k
    parent    = (j>jmin) & (j<jmax) & (r<rmax) #& (vclass==1)
    sel       = (parent) & (jw1>1.6*jk-1.6) & (jw1<1.6*jk-0.5) & (jw1>-2.5*jk+0.1) & (jw1<-0.5*jk+0.1)
    return sel



def get_4most_bg_new_sel(cat, mag_r_lim=19):
    
    not_in_gaia = cat['REF_CAT'] == '  '
    raw_mag_r = 22.5-2.5*np.log10(cat['FLUX_R'])
    
    mag_r = 22.5-2.5*np.log10(cat['FLUX_R']/cat['MW_TRANSMISSION_R'])
    mag_z = 22.5-2.5*np.log10(cat['FLUX_Z']/cat['MW_TRANSMISSION_Z'])
    sel = ((cat['GAIA_PHOT_G_MEAN_MAG'] - raw_mag_r) > 0.6) & ~not_in_gaia
    sel |= not_in_gaia

    #maskbit arround bright stars
    sel &= ~(cat['MASKBITS'] & 2**1 > 0) 
    #maskbit arround large galaxies
    sel &= ~(cat['MASKBITS'] & 2**12 > 0)
    sel &= ~(cat['MASKBITS'] & 2**13 > 0)

    #fiber mag cut
    fib_mag_r = 22.5-2.5*np.log10(cat['FIBERFLUX_R']/cat['MW_TRANSMISSION_R'])
    mask = mag_r < 17.8
    mask &= fib_mag_r < (22.9 + (mag_r - 17.8))
    mask1 = (mag_r > 17.8) & (mag_r < 20)
    mask1 &= fib_mag_r < 22.9
    sel &= (mask|mask1)

    # cut spurious objects
    mag_g = 22.5-2.5*np.log10(cat['FLUX_G']/cat['MW_TRANSMISSION_G'])
    mask = (mag_g-mag_r < -1) | (mag_g-mag_r > 4)
    mask |= (mag_r-mag_z < -1) | (mag_r-mag_z > 4)
    sel &= ~mask

    # cut according Nobs
    mask = (cat['NOBS_G'] == 0) | (cat['NOBS_R'] == 0) | (cat['NOBS_Z'] == 0)
    sel &= ~mask

    # fibertot mag cut
    fibtot_mag_r = 22.5-2.5*np.log10(cat['FIBERTOTFLUX_R']/cat['MW_TRANSMISSION_R'])
    mask = (mag_r > 12) & (fibtot_mag_r < 15) 
    sel &= ~mask

    # Additional photometric cut
    sel &= mag_r-mag_z < (0.9*(mag_g-mag_r)-0.2)
    
    # Additionnal cut on stars with parallax or pmra or pmdec !=0 and snr >3
    snr_par = np.abs(cat['PARALLAX']*np.sqrt(cat['PARALLAX_IVAR']))
    snr_pmra = np.abs(cat['PMRA']*np.sqrt(cat['PMRA_IVAR']))
    snr_pmdec = np.abs(cat['PMDEC']*np.sqrt(cat['PMDEC_IVAR']))
    mask = cat['PARALLAX'] != 0
    mask &= (snr_pmra > 3) | (snr_pmra > 3) | (snr_pmdec > 3) 
    sel &= ~mask
    # Mag cut
    sel &= mag_r < mag_r_lim
    return sel


def get_4most_lrg_new_sel(data, shift = 0.11): #shift = 0.0825

    gflux = data['FLUX_G']
    gMW = data['MW_TRANSMISSION_G']
    gmag = 22.5 - 2.5*np.log10(gflux/gMW)

    rflux = data['FLUX_R']
    rMW = data['MW_TRANSMISSION_R']
    rmag = 22.5 - 2.5*np.log10(rflux/rMW)

    zflux = data['FLUX_Z']
    zMW = data['MW_TRANSMISSION_Z']
    zmag = 22.5 - 2.5*np.log10(zflux/zMW)

    w1flux = data['FLUX_W1']
    w1MW = data['MW_TRANSMISSION_W1']
    w1mag = 22.5 - 2.5*np.log10(w1flux/w1MW)

    fiberz_flux = data['FIBERFLUX_Z']
    fiberz = 22.5 - 2.5*np.log10(fiberz_flux/zMW)

    alpha_new = 1.8
    beta_new = 17.14 - shift
    gamma_new = 16.33 - 1.5*shift

    mask_tot_ls = fiberz < 21.6
    mask_tot_ls &= (zmag - w1mag) > 0.8*(rmag - zmag) - 0.6
    mask_tot_ls &= ((gmag - w1mag) > 2.9) | ((rmag - w1mag) > 1.8)  # pq 2.97 et pas 2.9
    mask_tot_ls &= (rmag - w1mag > alpha_new*(w1mag - beta_new)) & ((rmag - w1mag > w1mag - gamma_new))

    # I would remove this cut
    #mask_tot_ls &= (photz > 0) & (photz < 1.4)

    # cut according Nobs,
    mask = (data['NOBS_G'] == 0) | (data['NOBS_R'] == 0) | (data['NOBS_Z'] == 0) | (data['NOBS_W1'] == 0)
    mask_tot_ls &= ~mask
    # cut according bad photometry
    mask = (data['FLUX_IVAR_R'] < 0) | (data['FLUX_IVAR_Z'] < 0) | (data['FLUX_IVAR_W1'] < 0) | (data['FLUX_G'] < 0)
    mask_tot_ls &= ~mask

    # Additionnal cuts from the DESI LRG TS
    mask_tot_ls &= fiberz > 17.5
    mask_tot_ls &= data['GAIA_PHOT_G_MEAN_MAG'] < 18 
    mask_tot_ls &= ~(data['MASKBITS'] & 2**1 > 0)
    mask_tot_ls &= ~(data['MASKBITS'] & 2**12 > 0)
    mask_tot_ls &= ~(data['MASKBITS'] & 2**13 > 0)
    
    # Additionnal cut on stars with parallax or pmra or pmdec !=0 and snr >3
    snr_par = np.abs(data['PARALLAX']*np.sqrt(data['PARALLAX_IVAR']))
    snr_pmra = np.abs(data['PMRA']*np.sqrt(data['PMRA_IVAR']))
    snr_pmdec = np.abs(data['PMDEC']*np.sqrt(data['PMDEC_IVAR']))
    mask = data['PARALLAX'] != 0
    mask &= (snr_par > 3) | (snr_pmra > 3) | (snr_pmdec > 3) 
    mask_tot_ls &= ~mask
    
    return mask_tot_ls
	

def get_desi_bright_bgs_sel(cat, mag_r_lim=19.5):
    
    not_in_gaia = cat['REF_CAT'] == '  '
    raw_mag_r = 22.5-2.5*np.log10(cat['FLUX_R'])
    mag_r = 22.5-2.5*np.log10(cat['FLUX_R']/cat['MW_TRANSMISSION_R'])

    mag_z = 22.5-2.5*np.log10(cat['FLUX_Z']/cat['MW_TRANSMISSION_Z'])
    sel = ((cat['GAIA_PHOT_G_MEAN_MAG'] - raw_mag_r) > 0.6) & ~not_in_gaia
    sel |= not_in_gaia

    #maskbit arround bright stars
    sel &= ~(cat['MASKBITS'] & 2**1 > 0) 

    #fiber mag cut
    fib_mag_r = 22.5-2.5*np.log10(cat['FIBERFLUX_R']/cat['MW_TRANSMISSION_R'])
    mask = mag_r < 17.8
    mask &= fib_mag_r < (22.9 + (mag_r - 17.8))
    mask1 = (mag_r > 17.8) & (mag_r < 20)
    mask1 &= fib_mag_r < 22.9
    sel &= (mask|mask1)

    # cut spurious objects
    mag_g = 22.5-2.5*np.log10(cat['FLUX_G']/cat['MW_TRANSMISSION_G'])
    mask = (mag_g-mag_r < -1) | (mag_g-mag_r > 4)
    mask |= (mag_r-mag_z < -1) | (mag_r-mag_z > 4)
    sel &= ~mask

    # cut according Nobs
    mask = (cat['NOBS_G'] == 0) | (cat['NOBS_R'] == 0) | (cat['NOBS_Z'] == 0)
    sel &= ~mask

    # fibertot mag cut
    fibtot_mag_r = 22.5-2.5*np.log10(cat['FIBERTOTFLUX_R']/cat['MW_TRANSMISSION_R'])
    mask = (mag_r > 12) & (fibtot_mag_r < 15) 
    sel &= ~mask

    sel &= mag_r < mag_r_lim
    return sel	
   
    
def projection_ra(ra, ra_center=0):
    """Shift `ra` to the origin of the Axes object and convert to radians.

    Parameters
    ----------
    ra : array-like
        Right Ascension in degrees.

    Returns
    -------
    array-like
        `ra` converted to plot coordinates.

    Notes
    -----
    In matplotlib, map projections expect longitude (RA), latitude (Dec)
    in radians with limits :math:`[-\pi, \pi]`, :math:`[-\pi/2, \pi/2]`,
    respectively.
    """
    #
    # Shift RA values.
    #
    r = np.remainder(ra + 360 - ra_center, 360)
    #
    # Scale conversion to [-180, 180].
    #
    r[r > 180] -= 360
    #
    # Reverse the scale: East to the left.
    #
    r = -r
    return np.radians(r)

def projection_dec(dec):
        """Shift `dec` to the origin of the Axes object and convert to radians.

        Parameters
        ----------
        dec : array-like
            Declination in degrees.

        Returns
        -------
        array-like
            `dec` converted to plot coordinates.
        """
        return np.radians(dec)

    
def get_skyaera(polygon, seed=42):
    '''
        Get sky aera for a given polygon 
    '''

    full_sky = 4*np.pi*(180/np.pi)**2
    ra = projection_ra(np.random.uniform(0,360, size=10000000))
    dec = projection_dec(np.random.uniform(-90,90, size=10000000))
    mask = polygon.contains_points(np.array([ra,dec]).T)
    return mask.sum()*full_sky/mask.size


def get_4most_s8foot(ra, dec, regions=['ngc', 'sgc'], if_deg=True, bg_fp=False, polygon_dir=None):
    '''
        Return mask where True values are position inside the 4most S8 footprint. Caveat: The initial polygons are rotated by 115 degrees.  
        
        Parameters
        ----------
        ra : array-like
            Right Ascension in radians or degrees.
        
        dec : array-like
            Declination in radians or degrees.
            
        (Optional) 
		bg_fp: bool, default: False
            Add additional cut to DEC < -20 for Bright galaxies
            
        reg: str or list,  default: ['ngc', 'sgc']
            Region to apply the mask, 'ngc' or 'sgc' only
            
        if_deg: bool,  default: True
            Set True if ra and dec are given in degrees.
        
        polygon_dir: str, default:'/pscratch/sd/a/arocher/4MOST'
            Path of the directory where polygons are stored

        Returns
        -------
        bool array-like
            mask with True values are position inside the polygon
    '''
    
    if polygon_dir is None: 
         polygon_dir = os.path.join(os.path.dirname(__file__), 'mask_fp')

    if if_deg:
        ra = projection_ra(ra, ra_center=115)
        dec = projection_dec(dec)
    mask = np.ones_like(ra) == 0
    regions = [regions] if isinstance(regions, str) else regions

    for reg in regions:
        polygon = Path(np.load(os.path.join(polygon_dir,f'4most_{reg}_newfootprint.npy'), allow_pickle=True))
        m = polygon.contains_points(np.array([ra,dec]).T)
        if reg =='sgc':
            m &= dec < np.radians(-20)
        if bg_fp & (reg =='ngc'):
            m &= dec < np.radians(-20)
        mask |= m
    mask &= dec < np.radians(-7.5)
    return mask


def getPointsOnSphere(nPoints):
    '''
        Function that give uniform randon sample of points on a sphere 
    '''
    u1, u2 = np.random.uniform(size=(2, nPoints))
    cmin = -1
    cmax = +1
    ra = 0 + u1*(2*np.pi-0)
    dec = np.pi - (np.arccos(cmin+u2*(cmax-cmin)))
    ur = np.zeros((nPoints, 3))
    ur[:, 0] = np.sin(dec) * np.cos(ra)
    ur[:, 1] = np.sin(dec) * np.sin(ra)
    ur[:, 2] = np.cos(dec)
    return ur


def random_point_on_sky(size):
    '''
        Function that give uniform randon sample of points on the full sky
    '''
    
    p = getPointsOnSphere(size).T
    r     = np.linalg.norm(p, axis=0)
    theta = 90 - (np.arccos(p[2] / r)    / np.pi * 180)            #theta and phi values in degrees
    phi   =       np.arctan(p[1] / p[0]) / np.pi * 180
    c     = SkyCoord(ra=phi, dec=theta, unit=(u.degree, u.degree)) #Create coordinate
    ra = np.random.uniform(0,360,size=len(p.T))
    return ra, c.dec.deg


def get_4most_skyaera(regions=['ngc', 'sgc'], bg_fp=False, polygon_dir=None, seed=42):
    '''
        Get sky aera for a given polygon 
    '''

    if polygon_dir is None:
         polygon_dir = os.path.join(os.path.dirname(__file__), 'mask_fp')
    np.random.seed(seed)
    full_sky = 4*np.pi*(180/np.pi)**2
    ra, dec = random_point_on_sky(10000000)
    regions = [regions] if isinstance(regions, str) else regions
    mask = []
    for reg in regions:
        mask += [get_4most_s8foot(ra, dec, regions=reg, bg_fp=bg_fp)]
        print(f'Area {reg}: {mask[-1].sum()*full_sky/mask[-1].size}')
    print(f'Total area: {np.concatenate(mask).sum()*full_sky/mask[0].size}')
    

# From A. Raichoor library

# https://desi.lbl.gov/svn/docs/technotes/targeting/target-truth/trunk/python/match_coord.py
# slightly edited (plot_q and keep_all_pairs removed; u => units)
def match_coord(ra1, dec1, ra2, dec2, search_radius=1., nthneighbor=1, verbose=True):
	'''
	Match objects in (ra2, dec2) to (ra1, dec1). 

	Inputs: 
		RA and Dec of two catalogs;
		search_radius: in arcsec;
		(Optional) keep_all_pairs: if true, then all matched pairs are kept; otherwise, if more than
		one object in t2 is match to the same object in t1 (i.e. double match), only the closest pair
		is kept.

	Outputs: 
		idx1, idx2: indices of matched objects in the two catalogs;
		d2d: distances (in arcsec);
		d_ra, d_dec: the differences (in arcsec) in RA and Dec; note that d_ra is the actual angular 
		separation;
	'''
	t1 = Table()
	t2 = Table()
	# protect the global variables   from being changed by np.sort
	ra1, dec1, ra2, dec2 = map(np.copy, [ra1, dec1, ra2, dec2])
	t1['ra'] = ra1
	t2['ra'] = ra2
	t1['dec'] = dec1
	t2['dec'] = dec2
	t1['id'] = np.arange(len(t1))
	t2['id'] = np.arange(len(t2))
	# Matching catalogs
	sky1 = SkyCoord(ra1*u.degree,dec1*u.degree, frame='icrs')
	sky2 = SkyCoord(ra2*u.degree,dec2*u.degree, frame='icrs')
	idx, d2d, d3d = sky2.match_to_catalog_sky(sky1, nthneighbor=nthneighbor)
	# This finds a match for each object in t2. Not all objects in t1 catalog are included in the result. 

	# convert distances to numpy array in arcsec
	d2d = np.array(d2d.to(u.arcsec))
	matchlist = d2d<search_radius
	if np.sum(matchlist)==0:
		if verbose:
			print('0 matches')
		return np.array([], dtype=int), np.array([], dtype=int), np.array([]), np.array([]), np.array([])
	t2['idx'] = idx
	t2['d2d'] = d2d	
	t2 = t2[matchlist]
	init_count = np.sum(matchlist)
	#--------------------------------removing doubly matched objects--------------------------------
	# if more than one object in t2 is matched to the same object in t1, keep only the closest match
	t2.sort('idx')
	i = 0
	while i<=len(t2)-2:
		if t2['idx'][i]>=0 and t2['idx'][i]==t2['idx'][i+1]:
			end = i+1
			while end+1<=len(t2)-1 and t2['idx'][i]==t2['idx'][end+1]:
				end = end+1
			findmin = np.argmin(t2['d2d'][i:end+1])
			for j in range(i,end+1):
				if j!=i+findmin:
					t2['idx'][j]=-99
			i = end+1
		else:
			i = i+1

	mask_match = t2['idx']>=0
	t2 = t2[mask_match]
	t2.sort('id')
	if verbose:
		print('Doubly matched objects = %d'%(init_count-len(t2)))
	# -----------------------------------------------------------------------------------------
	if verbose:
		print('Final matched objects = %d'%len(t2))
	# This rearranges t1 to match t2 by index.
	t1 = t1[t2['idx']]
	d_ra = (t2['ra']-t1['ra']) * 3600.    # in arcsec
	d_dec = (t2['dec']-t1['dec']) * 3600. # in arcsec
	##### Convert d_ra to actual arcsecs #####
	mask = d_ra > 180*3600
	d_ra[mask] = d_ra[mask] - 360.*3600
	mask = d_ra < -180*3600
	d_ra[mask] = d_ra[mask] + 360.*3600
	d_ra = d_ra * np.cos(t1['dec']/180*np.pi)
	##########################################
	return np.array(t1['id']), np.array(t2['id']), np.array(t2['d2d']), np.array(d_ra), np.array(d_dec)


# From regressis package E Chaussidon 2021


class Sagittarius(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit
    of the Sagittarius dwarf galaxy, as described in
        https://ui.adsabs.harvard.edu/abs/2003ApJ...599.1082M
    and further explained in
        https://www.stsci.edu/~dlaw/Sgr/.

    Parameters
    ----------
    representation : `~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    Lambda : `~astropy.coordinates.Angle`, optional, must be keyword
        The longitude-like angle corresponding to Sagittarius' orbit.
    Beta : `~astropy.coordinates.Angle`, optional, must be keyword
        The latitude-like angle corresponding to Sagittarius' orbit.
    distance : `~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.
    pm_Lambda_cosBeta : `~astropy.units.Quantity`, optional, must be keyword
        The proper motion along the stream in ``Lambda`` (including the
        ``cos(Beta)`` factor) for this object (``pm_Beta`` must also be given).
    pm_Beta : `~astropy.units.Quantity`, optional, must be keyword
        The proper motion in Declination for this object (``pm_ra_cosdec`` must
        also be given).
    radial_velocity : `~astropy.units.Quantity`, optional, keyword-only
        The radial velocity of this object.

    """

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'Lambda'),
            coord.RepresentationMapping('lat', 'Beta'),
            coord.RepresentationMapping('distance', 'distance')]
    }


def SGR_MATRIX():
    """Build the transformation matric from Galactic spherical to heliocentric Sgr coordinates based on Law & Majewski 2010."""
    SGR_PHI = (180 + 3.75) * u.degree  # Euler angles (from Law & Majewski 2010)
    SGR_THETA = (90 - 13.46) * u.degree
    SGR_PSI = (180 + 14.111534) * u.degree

    # Generate the rotation matrix using the x-convention (see Goldstein)
    D = rotation_matrix(SGR_PHI, "z")
    C = rotation_matrix(SGR_THETA, "x")
    B = rotation_matrix(SGR_PSI, "z")
    A = np.diag([1., 1., -1.])
    SGR_matrix = matrix_product(A, B, C, D)
    return SGR_matrix


@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, Sagittarius)
def galactic_to_sgr():
    """Compute the transformation matrix from Galactic spherical to heliocentric Sgr coordinates."""
    return SGR_MATRIX()


@frame_transform_graph.transform(coord.StaticMatrixTransform, Sagittarius, coord.Galactic)
def sgr_to_galactic():
    """Compute the transformation matrix from heliocentric Sgr coordinates to spherical Galactic."""
    return matrix_transpose(SGR_MATRIX())


def _get_galactic_plane(rot=120):
    """
    Galactic plane in ircs coordinates.

    Parameters
    ----------
    rot : float
        Rotation of the R.A. axis for sky visualisation plot. In DESI, it should be rot=120.

    Returns
    -------
    ra : float array
        Ordering (i.e. can use directly plt.plot(ra, dec with ls='-')) array containing R.A. values of the galactic plane in IRCS coordinates.
    dec : float array
        Ordering array containing Dec. values of the galactic plane in IRCS coordinates.
    """
    galactic_plane_tmp = SkyCoord(l=np.linspace(0, 2 * np.pi, 200) * u.radian, b=np.zeros(200) * u.radian, frame='galactic', distance=1 * u.Mpc)
    galactic_plane_icrs = galactic_plane_tmp.transform_to('icrs')

    ra, dec = galactic_plane_icrs.ra.degree - rot, galactic_plane_icrs.dec.degree
    ra[ra > 180] -= 360    # scale conversion to [-180, 180]
    ra = -ra               # reverse the scale: East to the left

    # get the correct order from ra=-180 to ra=180 after rotation
    index_galactic = np.argsort(galactic_plane_icrs.ra.wrap_at((180 + rot) * u.deg).degree)

    return ra[index_galactic], dec[index_galactic]


def _get_ecliptic_plane(rot=120):
    """ Same than _get_galactic_coordinates but for the ecliptic plane in IRCS coordiantes"""
    ecliptic_plane_tmp = SkyCoord(lon=np.linspace(0, 2 * np.pi, 200) * u.radian, lat=np.zeros(200) * u.radian, distance=1 * u.Mpc, frame='heliocentrictrueecliptic')
    ecliptic_plane_icrs = ecliptic_plane_tmp.transform_to('icrs')

    ra, dec = ecliptic_plane_icrs.ra.degree - rot, ecliptic_plane_icrs.dec.degree
    ra[ra > 180] -= 360    # scale conversion to [-180, 180]
    ra = -ra               # reverse the scale: East to the left

    index_ecliptic = np.argsort(ecliptic_plane_icrs.ra.wrap_at((180 + rot) * u.deg).degree)

    return ra[index_ecliptic], dec[index_ecliptic]


def _get_sgr_plane(rot=120):
    """ Same than _get_galactic_coordinates but for the Sagittarius Galactic plane in IRCS coordiantes"""
    sgr = coord.SkyCoord(Lambda=np.linspace(0, 2*np.pi, 128)*u.radian,
                     Beta=np.zeros(128)*u.radian, frame='sagittarius')
    sgr_plane_icrs = sgr.transform_to(coord.ICRS)
    
    ra, dec = sgr_plane_icrs.ra.degree - rot, sgr_plane_icrs.dec.degree
    ra[ra > 180] -= 360    # scale conversion to [-180, 180]
    ra = -ra               # reverse the scale: East to the left

    index_sgr = np.argsort(sgr_plane_icrs.ra.wrap_at((180 + rot) * u.deg).degree)

    return ra[index_sgr], dec[index_sgr]


def _get_sgr_stream(rot=120):
    """ Same than _get_galactic_coordinates but for the bottom and top line of the Sgr. Stream in IRCS coordiantes"""
    sgr_stream_top_tmp = coord.SkyCoord(Lambda=np.linspace(0, 2 * np.pi, 200) * u.radian, Beta=20 * np.pi / 180 * np.ones(200) * u.radian, distance=1 * u.Mpc, frame='sagittarius')
    sgr_stream_top_icrs = sgr_stream_top_tmp.transform_to(coord.ICRS)
    
    ra_top, dec_top = sgr_stream_top_icrs.ra.degree - rot, sgr_stream_top_icrs.dec.degree
    ra_top[ra_top > 180] -= 360    # scale conversion to [-180, 180]
    ra_top = -ra_top               # reverse the scale: East to the left

    index_sgr_top = np.argsort(sgr_stream_top_icrs.ra.wrap_at((180 + rot) * u.deg).degree)

    sgr_stream_bottom_tmp = coord.SkyCoord(Lambda=np.linspace(0, 2 * np.pi, 200) * u.radian, Beta=-15 * np.pi / 180 * np.ones(200) * u.radian, distance=1 * u.Mpc, frame='sagittarius')
    sgr_stream_bottom_icrs = sgr_stream_bottom_tmp.transform_to(coord.ICRS)

    ra_bottom, dec_bottom = sgr_stream_bottom_icrs.ra.degree - rot, sgr_stream_bottom_icrs.dec.degree
    ra_bottom[ra_bottom > 180] -= 360    # scale conversion to [-180, 180]
    ra_bottom = -ra_bottom               # reverse the scale: East to the left

    index_sgr_bottom = np.argsort(sgr_stream_bottom_icrs.ra.wrap_at((180 + rot) * u.deg).degree)

    return ra_bottom[index_sgr_bottom], dec_bottom[index_sgr_bottom], ra_top[index_sgr_top], dec_top[index_sgr_top]


def plot_moll(hmap, whmap=None, min=None, max=None, nest=False, title='', label=r'[$\#$ deg$^{-2}$]', filename=None, show=True, mask_dir=None, euclid_fp=False, stardens=False,
              galactic_plane=True, ecliptic_plane=False, sgr_plane=False, stream_plane=False, show_legend=True, fourmost_footprint=False, desi_footprint=False, qso_dr10_fp=False, atlas_fp=False, qso_fp=False,
              rot=115, projection='mollweide', figsize=(11.0, 7.0), xpad=.5, labelpad=5, xlabel_labelpad=10.0, ycb_pos=-0.05, cmap='RdYlBu_r', ticks=None, tick_labels=None):
    """
    From E. Chaussidon
    Plot an healpix map in nested scheme with a specific projection.

    Parameters
    ----------
    hmap : float array
        Healpix map 
    whmap : float array
        Weighted Healpix map 
    min : float
        Minimum value for the colorbar
    max : float
        Maximum value for the colorbar
    title : str
        Title for the figure. Title is just above the colorbar
    label : str
        Colobar label. Label is just on the right of the colorbar
    filename : str
        Path where the figure will be saved. If filename is not None, the figure is saved.
    show : bool
        If true display the figure
    galactic_plane / ecliptic_plane / sgr_plane / stream_plane : bool
        Display the corresponding plane on the figure.
    show_lengend : bool
        If True, display the legend corresponding to the plotted plane. A warning is raised if show_lengend is True and no plane is plotted.
    fourmost_footprint : bool
        If True, display 4most footprint
    desi_footprint : bool
        If True, display desi y5 footprint
    rot : float
        Rotation of the R.A. axis for sky visualisation plot. In DESI, it should be rot=120.
    projection : str
        Projection used to plot the map. In DESI, it should be mollweide
    figsize : float tuple
        Size of the figure
    xpad : float
        X position of label. Need to be adpated if figsize is modified.
    labelpad : float
        Y position of label. Need to be adpated if figsize is modified.
    xlabel_labelpad : float
        Position of the xlabel (R.A.). Need to be adpated if figsize is modified.
    ycb_pos : float
        Y position of the colorbar. Need to be adpated if figsize is modified or if title is too long.
    cmap : ColorMap class of matplotlib
        Usefull to adapt the color. Especially to create grey area for the Y5 footprint.
        For instance: cmap = plt.get_cmap('jet').copy()
                      cmap.set_extremes(under='darkgrey')  # --> everything under min will be darkgrey
    """
    # transform healpix map to 2d array
    plt.figure(1)
    m = hp.ma(hmap) if whmap is None else hp.ma(whmap/hmap)
    mask_2 = np.zeros(len(m))
    mask_2[m <= 0] = 1
    m.mask=mask_2
    map_to_plot = hp.cartview(m, nest=nest, rot=rot, flip='geo', fig=1, return_projected_map=True)
    plt.close()

    # build ra, dec meshgrid to plot 2d array
    ra_edge = np.linspace(-180, 180, map_to_plot.shape[1] + 1)
    dec_edge = np.linspace(-90, 90, map_to_plot.shape[0] + 1)

    ra_edge[ra_edge > 180] -= 360    # scale conversion to [-180, 180]
    ra_edge = -ra_edge               # reverse the scale: East to the left

    ra_grid, dec_grid = np.meshgrid(ra_edge, dec_edge)

    plt.figure(figsize=figsize)
    ax = plt.subplot(111, projection=projection)
    plt.subplots_adjust(left=0.14, bottom=0.18, right=0.96, top=0.90)

    mesh = plt.pcolormesh(np.radians(ra_grid), np.radians(dec_grid), map_to_plot, vmin=min, vmax=max, cmap=cmap, edgecolor='none', lw=0)

    if stardens:
        f_stardens = os.path.join(os.path.dirname(__file__), 'data', 'pixweight-dr10-128-new.fits')
        STARDENS = fitsio.FITS(f_stardens)[1]['STARDENS'][:]
        starmap_to_plot = hp.cartview(STARDENS, nest=True, rot=rot, flip='geo', fig=1, return_projected_map=True)
        ttt=ax.pcolormesh(np.radians(ra_grid), np.radians(dec_grid), starmap_to_plot, vmin=5000, vmax=30000, cmap='binary', edgecolor='none', lw=0)

    if mask_dir is None:
         mask_dir = os.path.join(os.path.dirname(__file__), 'mask_fp')

    if label is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_cb = inset_axes(ax, width="30%", height="4%", loc='lower left', bbox_to_anchor=(0.346, ycb_pos, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=0)
        cb = plt.colorbar(mesh, ax=ax, cax=ax_cb, orientation='horizontal', shrink=0.8, aspect=40, ticks=ticks)
        cb.outline.set_visible(False)
        cb.set_label(label, x=xpad, labelpad=labelpad)
        if tick_labels is not None:
            cb.ax.set_xticklabels(tick_labels)  # horizontal colorbar
        cb.ax.tick_params(size=0)

    if galactic_plane:
        ra, dec = _get_galactic_plane(rot=rot)
        ax.plot(np.radians(ra), np.radians(dec), linestyle='-', linewidth=0.8, color='black', label='Galactic plane')
    if ecliptic_plane:
        ra, dec = _get_ecliptic_plane(rot=rot)
        ax.plot(np.radians(ra), np.radians(dec), linestyle=':', linewidth=0.8, color='slategrey', label='Ecliptic plane')
    if sgr_plane:
        ra, dec = _get_sgr_plane(rot=rot)
        ax.plot(np.radians(ra), np.radians(dec), linestyle='--', linewidth=0.8, color='navy', label='Sgr. plane')
        if stream_plane:
            ra_bottom, dec_bottom, ra_top, dec_top = _get_sgr_stream(rot=rot)
            ax.plot(np.radians(ra), np.radians(dec), linestyle=':', linewidth=0.8, color='navy')
            ax.plot(np.radians(ra), np.radians(dec), linestyle=':', linewidth=0.8, color='navy')

    if fourmost_footprint:
        
        for reg in ['ngc','sgc']:
            pol = np.load(os.path.join(mask_dir, f'4most_{reg}_newfootprint.npy'), allow_pickle=True).T
            m=pol[1] < np.radians(-20)
            if reg == 'ngc':
                rra = np.linspace(0,360, 100)
                tt = projection_ra(rra, ra_center=115)
                tt = np.remainder(tt + np.pi*2, np.pi*2)
                tt[tt > np.pi] -= np.pi*2
                mm = (tt<pol[0][m].max()) & (tt>pol[0][m].min())
                ttt = tt[mm] - np.radians(115) + np.radians(rot)
                ttt = np.remainder(ttt + np.pi*2, np.pi*2)
                ttt[ttt > np.pi] -= np.pi*2
                ax.plot(ttt, projection_dec([-20]*100)[mm], lw=0.8, c='darkred', ls='--', zorder=10, label='BG cut')
            pol[0] -= np.radians(115)
            pol[0] += np.radians(rot)
            pol[0] = np.remainder(pol[0] + np.pi*2, np.pi*2)
            pol[0][pol[0] > np.pi] -= np.pi*2
            ax.plot(pol[0], pol[1], color="darkred", lw=2, zorder=10)
        ax.plot(pol[0], pol[1], color="darkred", lw=2, zorder=10, label='4MOST')

    if qso_dr10_fp:
        pol = np.load(os.path.join(mask_dir, 'qso_dr10_sgc_poly.npy'), allow_pickle=True).T 
        pol[0] += np.radians(rot)
        pol[0] = np.remainder(pol[0] + np.pi*2, np.pi*2)
        pol[0][pol[0] > np.pi] -= np.pi*2
        ax.plot(pol[0], pol[1], color="darkblue", lw=2, zorder=10, label='DR10')

    if atlas_fp:
        for reg in ['ngc','sgc']:
            pol = np.load(os.path.join(mask_dir, f'atlas_{reg}_poly.npy'), allow_pickle=True).T 
            pol[0] -= np.radians(115)
            pol[0] += np.radians(rot)
            pol[0] = np.remainder(pol[0] + np.pi*2, np.pi*2)
            pol[0][pol[0] > np.pi] -= np.pi*2
            ax.plot(pol[0], pol[1], color="steelblue", lw=2, zorder=10)
        ax.plot(pol[0], pol[1], color="steelblue", lw=2, zorder=10, label='ATLAS')

    if qso_fp:
        for name, rot_init in zip(['atlas_ngc_poly.npy', 'qso_sgc_poly.npy'], [115, 0]):
            pol = np.load(os.path.join(mask_dir, name), allow_pickle=True).T 
            pol[0] -= np.radians(rot_init)
            pol[0] += np.radians(rot)
            pol[0] = np.remainder(pol[0] + np.pi*2, np.pi*2)
            pol[0][pol[0] > np.pi] -= np.pi*2
            ax.plot(pol[0], pol[1], color="green", lw=2, zorder=10)
        ax.plot(pol[0], pol[1], color="green", lw=2, zorder=10, label='QSO')

    if  euclid_fp:
        for name in glob.glob(os.path.join(mask_dir, '*euclid*footprint*')):
            rot_init = 110 
            pol = np.load(name, allow_pickle=True).T 
            pol[0] -= np.radians(rot_init)
            pol[0] += np.radians(rot)
            pol[0] = np.remainder(pol[0] + np.pi*2, np.pi*2)
            pol[0][pol[0] > np.pi] -= np.pi*2
            ax.plot(pol[0], pol[1], color="darkorange", lw=2, zorder=10)
        ax.plot(pol[0], pol[1], color="darkorange", lw=2, zorder=10, label='Euclid')

        
    if desi_footprint:
        d = Table.read(os.path.join(mask_dir,"desi-14k-footprint-dark.ecsv"))
        for cap in ["NGC", "SGC"]:
            sel = d["CAP"] == cap
            _ = ax.plot(projection_ra(d["RA"][sel], ra_center=rot), projection_dec(d["DEC"][sel]), color="k", lw=2, zorder=10)
        ax.plot(projection_ra(d["RA"][sel], ra_center=rot), projection_dec(d["DEC"][sel]), color="k", lw=2, zorder=10, label='DESI Y5')


    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels + 360 + rot, 360)
    tick_labels = np.array([f'{lab}°' for lab in tick_labels])
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel('R.A. [deg]', labelpad=xlabel_labelpad)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Dec. [deg]')

    ax.grid(True)

    if show_legend:
        ax.legend(loc='upper right')
    if title:
        plt.title(title)
    if filename is not None:
        plt.savefig(filename, facecolor='w', bbox_inches='tight', pad_inches=0.1)
    if show:
        plt.show()
    else:
        plt.close()
        
def create_hp_map(ra, dec, nside=128, weight=None, lonlat=True, nest=False):
    pix = hp.ang2pix(nside, ra, dec, lonlat=lonlat, nest=nest)
    hp_map = np.bincount(pix, weights=weight, minlength=hp.nside2npix(nside)) * 1.0
    hp_map = hp_map / hp.nside2pixarea(nside, degrees=True)
    return hp_map


def healpix_in_sgc(nside,nest=False):
    theta, phi = hp.pix2ang(nside, range(hp.nside2npix(nside)),nest=nest)  
    ra= phi*180./np.pi
    dec = 90.-(theta*180./np.pi)
    mask_NGC = SkyCoord(ra,dec , frame='icrs', unit='deg').transform_to('galactic').b > 0
    return mask_NGC

def ra_dec_in_sgc(ra, dec, unit='deg'):
    return SkyCoord(ra, dec, frame='icrs', unit=unit).transform_to('galactic').b > 0




def run_sys_regression(regression_type, X_train, Y_train, X_eval, keep_to_train, nfold=6, use_kfold = False):
     if use_kfold: 
        return run_nfold_regression(regression_type, nfold, X_train, Y_train, keep_to_train, nside = 128)
     else:
        return run_regression(regression_type, X_train, Y_train, X_eval, normalize=True)

def run_regression(regression_type, X_train, Y_train, X_eval, normalize=True):
    regressor = LinearRegression() if regression_type.upper() == 'LINEAR' else RandomForestRegressor()
    normalize = True  if regression_type.upper() == 'LINEAR' else False
    X_for_training = X_train.copy()
    if normalize:
        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        X_for_training = (X_train - mean) / std
        X_eval = (X_eval - mean) / std
    regressor.fit(X_for_training, Y_train)
    Y_pred = regressor.predict(X_eval)
    return Y_pred


def build_kfold(kfold, pixels, groups):
        """
        Build the folding of pixels with scikit-learn's :class:`~sklearn.model_selection.GroupKfold` using a specific grouping given by group.
        All pixels in the same group will be in the same fold.

        Parameters
        ----------
        kfold : GroupKfold
            scikit-learn class with split method to build the group k-fold.
        pixels : array like
            List of pixels which must be splitted in k-fold.
        groups : array like
            Same size as pixels. It contains the group label for each pixel in pixels.
            All pixels in the same group will be in the same fold.

        Returns
        -------
        index : list of list
            For each fold, the index list of pixels belonging to that fold.
        """
        index = []
        for index_train, index_test in kfold.split(pixels, groups=groups):
            index += [index_test]
        return index


def run_nfold_regression(regression_type, nfold, X_train, Y_train, keep_to_train, nside = 128):

    size_group = 1000 * (nside / 256)**2  # define to have ~ 52 deg**2 for each patch (ie) group   

    kfold = GroupKFold(n_splits=nfold)
    pixels = np.arange(hp.nside2npix(nside))[keep_to_train]
    groups = [i // size_group for i in range(pixels.size)]
    index = build_kfold(kfold, pixels, groups)
    Y_predkfolf = np.zeros(pixels.size)

    for i in range(nfold):
        fold_index = index[i]
        # select row for the training i.e. remove fold index
        X_fold, Y_fold = np.delete(X_train, fold_index, axis=0), np.delete(Y_train, fold_index)
        # select row for the evaluation
        X_eval_fold = X_train[fold_index]
        print(f'nflod {i} is running')
        Y_predkfolf[fold_index] = run_regression(regression_type, X_fold, Y_fold, X_eval_fold, normalize=None)
        print(f'nflod {i} done')
    return Y_predkfolf


def apply_mask_to_hpmap(hpmap, mask):
    hpmap_mask = hpmap.copy()
    hpmap_mask[~mask] = 0
    return hpmap_mask



def get_wmap(targets, features_pixmap, mask_pixmap, tracer, regions=['DECALS', 'DES'], regression_type = 'Linear', nfold = 6, use_kfold = False,
             feature_names = ['EBV', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']):
    
    if regression_type.upper() =='RF':
        use_kfold=True
    w = np.zeros_like(targets)
    if isinstance(regions, str):
        regions=[regions]

    if tracer=='LRG':
        feature_names = ['EBV', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'PSFDEPTH_W1', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']
    elif tracer == 'BG':
        feature_names = ['STARDENS', 'EBV', 'GALDEPTH_R', 'GALDEPTH_Z', 'GALDEPTH_G', 'PSFSIZE_R', 'PSFSIZE_G', 'PSFSIZE_Z']

    for reg in regions:
        if reg.upper() == 'DECALS':
            region = ~mask_pixmap['ISDES'] & mask_pixmap[f'IS4MOST_{tracer}']
        elif reg.upper() == 'DES':
            region = mask_pixmap['ISDES']

        fracarea = features_pixmap['FRACAREA_12290']

        keep_to_train = (fracarea > np.quantile(fracarea[fracarea>0], q=0.05)) & (targets > 0) & region 

        region_to_pred = keep_to_train & mask_pixmap[f'IS4MOST_{tracer}']

        #utils.plot_moll(utils.apply_mask_to_hpmap(targets, mask=keep_to_train), rot=115, min=300, max=600, desi_footprint=False, fourmost_footprint=True, label='deg-2', nest=True)

        normalized_targets = np.zeros_like(targets)
        mean_targets_density_estimators = np.mean(targets[keep_to_train] / fracarea[keep_to_train])
        normalized_targets[keep_to_train] = targets[keep_to_train] / (fracarea[keep_to_train] * mean_targets_density_estimators)

        #utils.plot_moll(normalized_targets, rot=115, min=0.5, max=1.5, desi_footprint=False, fourmost_footprint=True, label='deg-2', nest=True)


        X = np.vstack([features_pixmap[v] for v in feature_names]).T
        Y = normalized_targets
        X_train, Y_train, X_eval = X[keep_to_train], Y[keep_to_train], X[region_to_pred]

        Y_pred = np.zeros_like(targets)

        fin_mask = keep_to_train if use_kfold else region_to_pred
        Y_pred[fin_mask] = run_sys_regression(regression_type, X_train, Y_train, X_eval, keep_to_train, nfold=nfold, use_kfold = use_kfold)

        w[region_to_pred] = 1.0 / Y_pred[region_to_pred]
    return w

def get_features(tracer):
    if tracer=='LRG':
        feature_names = ['EBV', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'PSFDEPTH_W1', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']
    elif tracer == 'BG':
        feature_names = ['STARDENS', 'EBV', 'GALDEPTH_R', 'GALDEPTH_Z', 'GALDEPTH_G', 'PSFSIZE_R', 'PSFSIZE_G', 'PSFSIZE_Z']
    else:
        raise(f'{tracer} not known only LRG or BG')
    return feature_names


_all_feature_names = ['STARDENS', 'EBV', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1',
                      'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z']
_all_feature_label = [r'$\log_{10}$(Stellar Density)', 'E(B-V)', r'PSF Depth in $g$-band', r'PSF Depth in $r$-band', r'PSF Depth in $z$-band', r'PSF Depth in $W1$-band', 
                  r'PSF Size in $g$-band', r'PSF Size in $r$-band', r'PSF Size in $z$-band', r'Gal Depth in $g$-band', r'Gal Depth in $r$-band', r'Gal Depth in $z$-band']
_all_feature_labels_dict = dict(zip(_all_feature_names, _all_feature_label))


def plot_systmematics(targets_map, features_pixmap, feature_names=_all_feature_names, labels_map=None, ylim=0.2, nb_rows=3, fig_title=None, savename=None):
    
    nb_cols = (len(feature_names)+(nb_rows-1))//nb_rows
    fig,axx = plt.subplots(nb_rows,nb_cols,figsize=(nb_cols*4,nb_rows*3), sharey=True)
    axes=axx.flatten()
    if fig_title:
        fig.suptitle(fig_title, fontsize=15)

    if not isinstance(targets_map,list):
        targets_map = [targets_map]
    if not isinstance(feature_names,list):
        feature_names = [feature_names]
    
    if labels_map is None:
        labels_map = [None]*len(targets_map)
    if not isinstance(labels_map,list):
        labels_map = [labels_map]


    for feature, ax in zip(feature_names, axes):
        for (i, tar), label in zip(enumerate(targets_map), labels_map):
            nbins=25
            #keep_tot = hpmap != 0.0
            feature_map = features_pixmap[feature].copy()
            keep_tot = (tar!=0) & (feature_map!=0)
            feature_map = feature_map[keep_tot]

            #n_bar = np.mean(hpmap[keep_tot])
            n_bar =np.mean(tar[keep_tot])
            
            if 'DEPTH' in feature: 
                feature_map =-2.5*(np.log10(5/np.sqrt(feature_map))-9)
            if 'STARDENS' in feature:
                feature_map = np.log10(feature_map)
            
            boundary = np.quantile(feature_map, q=[0.02,0.98])
            counts, bins = np.histogram(feature_map, bins=nbins, range=boundary)
            cbin = 0.5*(bins[1:]+bins[:-1])
            keep_all = [np.all([feature_map>=bins[i], feature_map<bins[i+1]], axis=0) for i in range(nbins)]
            mean, std = np.array([np.mean((tar[keep_tot][mm]/n_bar)[tar[keep_tot][mm]/n_bar>0.1]) for mm in keep_all]), [np.std(tar[keep_tot][mm]/n_bar)/np.sqrt(mm.sum()) for mm in keep_all]
            
                

            ax.errorbar(cbin, mean/mean.mean() -1, np.hstack(std), fmt='.', ls='-', color=f'C{i}', label=label)
            ax.axhline(0, ls='--', lw=1.2, c='k')
            ax.grid()
            ax.set_xlabel(_all_feature_labels_dict[feature], fontsize=12)
            ax.set_ylim(-ylim,ylim)
            ax.grid(True)

            normalisation = counts.sum()
            x_hist, y_hist = np.ones(2*nbins), np.ones(2*nbins)
            x_hist[::2], x_hist[1::2] = bins[:-1], bins[1:]
            y_hist[::2], y_hist[1::2] = counts, counts
            shift = -ylim
            ax.plot(x_hist, y_hist/normalisation+shift, color=f'C{i}')
            ax.fill_between(x_hist, y_hist/normalisation+shift, y2=shift, alpha=0.2, color=f'C{i}')

    [axx[i][0].set_ylabel(r'$\frac{n-\bar n}{\bar n}$', fontsize=12) for i in range(len(axx))]
    if labels_map[0] is not None : axx[0][0].legend(fontsize=12)
    fig.tight_layout()
    if savename is not None:
        fig.savefig(savename)
    fig.show()

def get_weight_from_wmap(ra, dec, wmap, nest=True, lonlat=True):
    nside = hp.npix2nside(wmap.size)
    ipix = hp.ang2pix(nside, ra, dec, nest=nest, lonlat=lonlat)
    return wmap[ipix]


def plot_kfold(keep_to_train, nside=128, nfold=6):
    
    size_group = 1000 * (nside / 256)**2  # define to have ~ 52 deg**2 for each patch (ie) group   
    kfold = GroupKFold(n_splits=nfold)
    pixels = np.arange(hp.nside2npix(nside))[keep_to_train]
    groups = [i // size_group for i in range(pixels.size)]
    index = build_kfold(kfold, pixels, groups)
    Y_predkfolf = np.zeros(pixels.size)

    for i in range(nfold):
        fold_index = index[i]
        Y_predkfolf[fold_index] = i+1
    kfoldmap = np.zeros(hp.nside2npix(nside))
    kfoldmap[keep_to_train]= Y_predkfolf
    plot_moll(kfoldmap, nest=True, min=0, max=nfold)