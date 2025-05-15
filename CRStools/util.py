"""
Utility functions and classes for angular clustering pipeline.

This module provides classes and functions for:
- Operating on pixel masks used for astronomical masks.
- Performing cosmology lookups.
- Applying various selection masks.
- Retrieving built-in cosmology models from Astropy.
"""

from dataclasses import dataclass
import numpy as np
from numpy.polynomial import Polynomial
from numpy.random import default_rng
from typing import Any
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.cosmology import default_cosmology
from astropy.table import Table
import healpy as hp
import astropy.cosmology.units as cu
import pandas as pd


class PixMask:
    """
    Class for handling pixel mask operations.

    This class opens a FITS file containing a pixel mask and provides methods to 
    check if coordinates fall in a masked (or unmasked) area and to apply the mask 
    to catalogues.
    """

    def __init__(self, mask_file, data_column=1):
        """
        Initialize the PixMask object by loading the mask file.

        Parameters:
            mask_file (str): File path to the FITS file containing the pixel mask.
            data_column (int, optional): Index of the HDU to use from the FITS file.
                                         Defaults to 1.
        """
        self.hdu = fits.open(mask_file)[data_column]
        self.wcs = WCS(self.hdu.header)

    def check_mask(self):
        """
        Interactively check the mask at given celestial coordinates.

        This function prints the header and dimension of the data,
        then repeatedly prompts the user for RA and Dec in degrees.
        For each coordinate, it prints the corresponding mask value or indicates if
        the coordinate is outside the mask.

        Returns:
            None
        """
        print(self.hdu.header, self.hdu.data.shape)

        ans = 'xxx'
        while ans != '':
            ans = input('RA, Dec [deg]: ')
            if ans != '':
                coords = SkyCoord(
                    *[float(c) for c in ans.split(',')], frame='icrs', unit='deg')
                x, y = self.wcs.world_to_pixel(coords)
                try:
                    print(coords, x, y, self.hdu.data[int(y), int(x)])
                except IndexError:
                    print(coords, x, y, 'Outside mask')

    def apply_mask(self, ra, dec):
        """
        Apply the pixel mask to an array of catalogue coordinates.

        Computes the pixel positions from the provided RA and Dec arrays and retrieves 
        the corresponding mask values. For coordinates outside the mask bounds, returns -1.

        Parameters:
            ra (array_like): Array of right ascension values in degrees.
            dec (array_like): Array of declination values in degrees.

        Returns:
            numpy.ndarray: Array of mask values corresponding to the input coordinates.
                           Returns -1 for positions outside the mask.
        """
        coords = SkyCoord(ra.astype(np.float64), dec.astype(np.float64),
                          frame='icrs', unit='deg')
        x, y = self.wcs.world_to_pixel(coords)
        ix, iy = x.astype(int), y.astype(int)
        inside = (ix >= 0) * (ix <
                              self.hdu.data.shape[1]) * (iy >= 0) * (iy < self.hdu.data.shape[0])
        mask_array = np.zeros(len(ix), dtype=int)
        mask_array[inside] = self.hdu.data[iy[inside], ix[inside]]
        mask_array[~inside] = -1

        return mask_array


class CosmoLookup(object):
    """
    Class for cosmological distance and volume lookups.

    This class precomputes various cosmological quantities over a range of redshifts and 
    provides interpolation methods to retrieve these quantities for given redshift or 
    comoving distance values.
    """

    def __init__(self, cosmo, zbins=np.linspace(0.01, 1, 100)):
        """
        Initialize the CosmoLookup object.

        Parameters:
            cosmo (astropy.cosmology.FLRW): Cosmology object.
            zbins (array_like, optional): Array of redshift bin edges. Defaults to 100 bins 
                                          linearly spaced from 0.01 to 1.
        """
        self.cosmo = cosmo
        self._z = zbins
        self._nz = len(self._z)
        self._x = cosmo.comoving_distance(self._z).value
        self._distmod = cosmo.distmod(self._z).value
        self._comoving_volume = cosmo.comoving_volume(self._z).value
        self._differential_comoving_volume = cosmo.differential_comoving_volume(
            self._z).value
        self._dxdz = np.gradient(self._x, zbins[1] - zbins[0])
        self.xmin, self.xmax = self._x[0], self._x[-1]

    def z_at_dist(self, x):
        """
        Returns the redshift corresponding to a given comoving distance.

        Parameters:
            x (float or array_like): Comoving distance value(s).

        Returns:
            float or numpy.ndarray: Interpolated redshift(s).
        """
        return np.interp(x, self._x, self._z)

    def dc(self, z):
        """
        Returns the comoving distance for a given redshift.

        Parameters:
            z (float or array_like): Redshift value(s).

        Returns:
            float or numpy.ndarray: Interpolated comoving distance(s).
        """
        return np.interp(z, self._z, self._x)

    def dxdz(self, z):
        """
        Get the differential of comoving distance with respect to redshift.

        Parameters:
            z (float or array_like): Redshift value(s).

        Returns:
            float or numpy.ndarray: Interpolated derivative d(comoving distance)/dz.
        """
        return np.interp(z, self._z, self._dxdz)

    def distmod(self, z):
        """
        Get the distance modulus for a given redshift.

        Parameters:
            z (float or array_like): Redshift value(s).

        Returns:
            float or numpy.ndarray: Interpolated distance modulus(es).
        """
        return np.interp(z, self._z, self._distmod)

    def comoving_volume(self, z):
        """
        Get the comoving volume up to a given redshift.

        Parameters:
            z (float or array_like): Redshift value(s).

        Returns:
            float or numpy.ndarray: Interpolated comoving volume(s).
        """
        return np.interp(z, self._z, self._comoving_volume)

    def differential_comoving_volume(self, z):
        """
        Get the differential comoving volume at a given redshift.

        Parameters:
            z (float or array_like): Redshift value(s).

        Returns:
            float or numpy.ndarray: Interpolated differential comoving volume(s).
        """
        return np.interp(z, self._z, self._differential_comoving_volume)


def ra_shift(ra):
    """
    Rotate right ascension values to avoid the discontinuity at RA=0.

    This function shifts RA values by subtracting 180 degrees, wrapping around any
    negative values by adding 360 degrees.

    Parameters:
        ra (array_like): Array of right ascension values in degrees.

    Returns:
        numpy.ndarray: Adjusted right ascension values.
    """
    ras = np.array(ra) - 180
    ras[ras < 0] += 360
    return ras


def selection_mask(ra, dec, field_limits, pixel_mask):
    """
    Apply selection based on field limits and a pixel mask to catalogue coordinates.

    The selection criteria include:
      - RA and Dec within the specified field limits.
      - The pixel mask value must be 0 (indicating an unmasked region).

    Parameters:
        ra (array_like): Array of right ascension values in degrees.
        dec (array_like): Array of declination values in degrees.
        field_limits (list): Field limits as [ra_min, ra_max, dec_min, dec_max].
        pixel_mask (PixMask): PixMask object with an apply_mask method.

    Returns:
        numpy.ndarray: Boolean array indicating which entries pass the selection.
    """
    ra_min, ra_max, dec_min, dec_max = field_limits
    sel = ((ra - ra_min) % 360 <= (ra_max - ra_min) % 360) & \
          (dec_min <= dec) & (dec <= dec_max) & \
          (pixel_mask.apply_mask(ra, dec) == 0)
    return sel


def LS_masking(data, selections=None):
    """
    Apply Legacy Survey (LS) masking on the dataset based on specified selection bits.

    This function creates a boolean mask based on the MASKBITS and WISEMASK_W1 bits provided 
    in the selections dictionary. It iterates through given bits and masks the data accordingly.

    Parameters:
        data (numpy.recarray or structured array): Input data containing mask columns.
        selections (dict, optional): Dictionary with keys 'MASKBITS' and 'WISEMASK_W1' that list 
                                     the bit positions to check. Defaults to None.

    Returns:
        numpy.ndarray: Boolean mask array where True indicates unmasked (selected) data.
    """
    MASKBITS = selections['MASKBITS']
    WISEMASK_W1 = selections['WISEMASK_W1']
    print('MASKBITS, WISEMASK_W1', MASKBITS, WISEMASK_W1, flush=True)
    mask = np.ones(len(data), dtype=bool)

    if len(MASKBITS) > 0:
        print('MASKBITS', flush=True)
        for bit in MASKBITS:
            print('bit', bit, flush=True)
            mask &= ~(data['MASKBITS'] & 2 ** bit > 0)

    if len(WISEMASK_W1) > 0:
        print('WISEMASK_W1', flush=True)
        for bit in WISEMASK_W1:
            print('bit', bit, flush=True)
            mask &= ~(data['WISEMASK_W1'] & 2 ** bit > 0)
    return mask


def LS_masking_BG(data, selections=None):
    """
    Apply Legacy Survey masking for BG data.

    For BG data, only the MASKBITS are applied, ignoring the WISEMASK_W1 mask.

    Parameters:
        data (numpy.recarray or structured array): Input data containing the MASKBITS column.
        selections (dict, optional): Dictionary with key 'MASKBITS'. Defaults to None.

    Returns:
        numpy.ndarray: Boolean mask array for BG data.
    """
    MASKBITS = selections['MASKBITS']
    print('MASKBITS', MASKBITS, flush=True)
    mask = np.ones(len(data), dtype=bool)

    if len(MASKBITS) > 0:
        print('MASKBITS', flush=True)
        for bit in MASKBITS:
            print('bit', bit, flush=True)
            mask &= ~(data['MASKBITS'] & 2 ** bit > 0)
    return mask


def LS_final_masking(data, selections=None):
    """
    Apply final Legacy Survey masking based on the data type selection.

    Different masking criteria are applied depending on the selection type:
      - 'BG' or 's0801': Uses a set of MASKBITS.
      - 'LRG' or 's0802': Uses MASKBITS and also filters based on WISEMASK_W1.
      - Otherwise, applies a default MASKBITS mask.

    Parameters:
        data (numpy.recarray or structured array): Input data containing MASKBITS and possibly WISEMASK_W1.
        selections (str, optional): A string indicating the type of masking to apply.
                                    Defaults to None.

    Returns:
        numpy.ndarray: Boolean mask array after final masking.
    """
    if selections == 'BG' or selections == 's0801':
        mask = ~(data['MASKBITS'] & 2 ** 1 > 0)
        mask &= ~(data['MASKBITS'] & 2 ** 12 > 0)
        mask &= ~(data['MASKBITS'] & 2 ** 13 > 0)
        mask &= ~(data['MASKBITS'] & 2 ** 11 > 0)

    elif selections == 'LRG' or selections == 's0802':
        mask = ~(data['MASKBITS'] & 2 ** 1 > 0)
        mask &= ~(data['MASKBITS'] & 2 ** 12 > 0)
        mask &= ~(data['MASKBITS'] & 2 ** 13 > 0)
        mask &= ~(data['MASKBITS'] & 2 ** 11 > 0)
        mask &= ~(data['WISEMASK_W1'] > 0)
    else:
        mask = ~(data['MASKBITS'] & 2 ** 1 > 0)
    return mask


def get_builtin_cosmology(model_name=None):
    """
    Retrieve a built-in cosmology model from Astropy.

    This function sets the default cosmology to the specified model name
    using Astropy's default_cosmology module, then retrieves the cosmology object.

    Parameters:
        model_name (str): Name of the cosmological model to use.

    Returns:
        astropy.cosmology.FLRW: The resulting cosmology object.
    """
    if model_name is None:
        model_name = 'Planck18'

    with default_cosmology.set(model_name):
        # Set the default cosmology to the specified model
        pass
    cosmo = default_cosmology.get()

    return cosmo


def get_n_z_mag_slises(redshifts, mags, z_bins, mag_bins, normalise=False):
    """
    Compute the redshift distribution n(z) for different magnitude slices.
    This function uses the `get_n_z` function to compute n(z) for each magnitude slice
    defined by the provided magnitude bins.
    The redshift distribution is computed for the redshifts corresponding to each
    magnitude slice.
    Parameters:
        redshifts (array_like): Array of redshift values.
        mags (array_like): Array of magnitude values.
        z_bins (array_like): Array of bin edges for the redshift histogram.
        mag_bins (array_like): Array of bin edges for the magnitude histogram.
        normalise (bool, optional): If True, normalize the histogram to sum to 1.
                                    Defaults to False.
    Returns:
        dict: Dictionary where keys are indices of magnitude slices and values are
              the corresponding n(z) distributions for those slices.
    """

    nz_mag_slices = {}
    for imag in range(len(mag_bins) - 1):
        mag_min = mag_bins[imag]
        mag_max = mag_bins[imag + 1]
        sel = (mags >= mag_min) & (mags < mag_max)
        nz_mag_slices[imag] = get_n_z(
            redshifts[sel], z_bins, normalise=normalise)

    nz_mag_slices['mag_bins'] = mag_bins
    nz_mag_slices['z_bins'] = z_bins
    nz_mag_centres = 0.5 * (mag_bins[:-1] + mag_bins[1:])
    nz_mag_slices['mag_centres'] = nz_mag_centres

    nz_mag_slices['z_centres'] = 0.5 * (z_bins[:-1] + z_bins[1:])
    nz_mag_slices['z_bins'] = z_bins

    return nz_mag_slices


def get_n_z(redshifts, z_bins):
    """
    Compute the redshift distribution n(z) from a given array of redshifts.
    This function uses a histogram to count the number of occurrences of redshift values
    within specified bins. The resulting histogram can be normalized to represent a
    probability distribution.

    Parameters:
        redshifts (array_like): Array of redshift values.
        z_bins (array_like): Array of bin edges for the histogram.
        normalise (bool, optional): If True, normalize the histogram to sum to 1.
                                    Defaults to False.

    Returns:
        numpy.ndarray: Histogram of redshift distribution n(z) for the specified bins.
    """
    nz, _ = np.histogram(redshifts, bins=z_bins)

    return nz


def get_dndV(redshifts, z_bins, survey_area, cosmo=None):
    """
    Compute the differential number density dN/dV for a given redshift distribution.
    This function calculates the number density of objects per unit volume
    in each redshift bin, taking into account the comoving volume of the shell
    defined by the redshift bin edges and the survey area.

    Parameters:
        redshifts (array_like): Array of redshift values.
        z_bins (array_like): Array of bin edges for the histogram.
        survey_area (float): Area of the survey in square degrees.
        cosmo (astropy.cosmology.FLRW, optional): Cosmology object. If None, uses Planck18.
    Returns:
        numpy.ndarray: Differential number density dN/dV for each redshift bin.
    """
    nz = get_n_z(redshifts, z_bins)
    dndV = np.zeros(len(z_bins) - 1)

    for iz in range(len(z_bins) - 1):
        N = nz[iz]
        z_min, z_max = z_bins[iz], z_bins[iz + 1]
        v_shell = comoving_volume_bin(z_min, z_max, survey_area, cosmo=cosmo)
        if v_shell > 0:
            dndV[iz] = N / v_shell
        else:
            dndV[iz] = 0.0

    return dndV


def get_dndz(redshifts, z_bins):
    """
    Compute the differential number density dN/dz for a given redshift distribution.
    This function calculates the number density of objects per unit redshift
    in each redshift bin, taking into account the width of the redshift bins.
    Parameters:
        redshifts (array_like): Array of redshift values.
        z_bins (array_like): Array of bin edges for the histogram.
    Returns:
        numpy.ndarray: Differential number density dN/dz for each redshift bin.
    """

    Nz = get_n_z(redshifts, z_bins)
    delta_z = z_bins[1:] - z_bins[:-1]
    Nz = Nz / delta_z
    return Nz


def survey_area(ra, dec, nside=256):
    """
    Calculate the survey area in square degrees given RA and Dec coordinates and a HEALPix nside.

    Parameters:
        ra (array_like): Array of right ascension values in degrees.
        dec (array_like): Array of declination values in degrees.
        nside (int): HEALPix nside parameter. Defaults to 256.

    Returns:
        float: Survey area in square degrees.
    """
    theta = np.radians(90 - np.array(dec))
    phi = np.radians(np.array(ra))

    pixels = hp.ang2pix(nside, theta, phi)
    unique_pixels = np.unique(pixels)

    area_per_pixel = hp.nside2pixarea(nside, degrees=True)
    return len(unique_pixels) * area_per_pixel


def comoving_volume_bin(z_min, z_max, area_sq_deg, cosmo=None):
    """
    Calculate the comoving volume of a shell between two redshift bins.
    The volume is computed using the comoving distance and the area of the shell
    in square degrees. The result is returned in (Mpc/h)^3.
    Parameters:
        z_min (float): Minimum redshift of the shell.
        z_max (float): Maximum redshift of the shell.
        area_sq_deg (float): Area of the shell in square degrees.
        cosmo (astropy.cosmology.FLRW, optional): Cosmology object. If None, uses Planck18.
    Returns:
        float: Comoving volume of the shell in (Mpc/h)^3.
    """

    if z_min < 0 or z_max < z_min:
        # Return 0 volume for invalid bins instead of raising an error,
        # as histogram might produce empty bins at edges.
        return 0.0

    if area_sq_deg <= 0:
        raise ValueError("Area must be positive")

    if cosmo is None:
        cosmo = get_builtin_cosmology(model_name='Planck18')
    # Area in steradians
    area_sr = area_sq_deg * (np.pi / 180)**2

    # Comoving volume out to z_max and z_min in Mpc^3
    # Use astropy units for clarity
    vol_max = cosmo.comoving_volume(z_max)  # Mpc^3
    vol_min = cosmo.comoving_volume(
        z_min) if z_min > 0 else 0 * cu.Mpc**3  # Mpc^3

    # Volume of the shell in Mpc^3, scaled by sky fraction
    v_shell_mpc3 = (vol_max - vol_min) * (area_sr / (4 * np.pi))

    # Convert to (Mpc/h)^3
    v_shell_mpch3 = v_shell_mpc3 * (cosmo.h**3)

    # Return the value in (Mpc/h)^3
    return v_shell_mpch3.to_value((cu.Mpc / cosmo.h)**3)


def get_n_z_table(redshifts, z_bins, cosmo=None, survey_area=None,
                  comov_V_norm=False, dndz=False, out_file=None):
    """
    Compute the redshift distribution n(z) from a given array of redshifts.
    This function uses a histogram to count the number of occurrences of redshift values
    within specified bins. The resulting histogram can be normalized to represent a
    probability distribution.

    Parameters:
        redshifts (array_like): Array of redshift values.
        z_bins (array_like): Array of bin edges for the histogram.
        normalise (bool, optional): If True, normalize the histogram to sum to 1.
                                    Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame containing the redshift distribution n(z) for the specified bins.

        
    """
    nz = get_n_z(redshifts, z_bins)

    z_min = z_bins[:-1]
    z_max = z_bins[1:]
    z_centre = 0.5 * (z_min + z_max)

    nz_table = pd.DataFrame({
        'z_min': z_min,
        'z_max': z_max,
        'z_centre': z_centre,
        'N_z': nz
    })

    if comov_V_norm:
        if survey_area is None:
            raise ValueError(
                "Survey area must be provided for comoving volume normalization.")
        if cosmo is None:
            print("Cosmology not provided, using default Planck18.", flush=True)
        nz_comov = comoving_volume_bin(z_min, z_max, survey_area, cosmo=cosmo)
        nz_table['nz_comov'] = nz / nz_comov

    if dndz:
        dndz = get_dndz(redshifts, z_bins)
        nz_table['dndz'] = dndz

    if out_file is not None:
        if out_file.endswith('.csv'):
            nz_table.to_csv(out_file, index=False)

        elif out_file.endswith('.fits'):
            table = Table.from_pandas(nz_table)
            table.write(out_file, format='fits', overwrite=True)

    return nz_table


def flux_to_mag_LS(flux, flux_mw_transmission=None):

    if flux_mw_transmission is None:
        mag = 22.5 - 2.5 * np.log10(flux)
    else:
        mag = 22.5 - 2.5 * np.log10(flux / flux_mw_transmission)

    return mag


def crs_bg_selection(cat, mag_r_lim=19.25, fib_mag_r_col='fib_mag_r', mag_r_col='mag_r',
                     mag_g_col='mag_g', mag_z_col='mag_z', fibtotal_mag_col='fibtot_mag_r',
                     raw_mag_r_col='raw_mag_r', if_mask: bool = False, r_z_cut_ratio=0.9,
                     r_z_cut=0.2, colour_cut_v2: bool = True):

    """
    Apply selection criteria to a catalogue of astronomical objects.
    The selection is based on various magnitude and colour cuts, as well as
    additional criteria such as the presence of Gaia data and the number of observations.
    The function also applies a mask to the data if specified.
    Parameters:
        cat (astropy.table.Table): Input catalogue containing astronomical data.
        mag_r_lim (float): Magnitude limit for the r-band. Default is 19.25.
        fib_mag_r_col (str): Column name for fiber magnitude in r-band. Default is 'fib_mag_r'.
        mag_r_col (str): Column name for magnitude in r-band. Default is 'mag_r'.
        mag_g_col (str): Column name for magnitude in g-band. Default is 'mag_g'.
        mag_z_col (str): Column name for magnitude in z-band. Default is 'mag_z'.
        fibtotal_mag_col (str): Column name for total fiber magnitude in r-band. Default is 'fibtot_mag_r'.
        raw_mag_r_col (str): Column name for raw magnitude in r-band. Default is 'raw_mag_r'.
        if_mask (bool): If True, apply the LS final masking. Default is False.
        r_z_cut_ratio (float): Ratio for the r-z colour cut. Default is 0.9.
        r_z_cut (float): Cut value for the r-z colour cut. Default is 0.2.
        colour_cut_v2 (bool): If True, apply the new colour cut. Default is True.
    Returns:
        numpy.ndarray: Boolean mask array indicating which entries pass the selection.
    """
    

    if mag_r_col in cat.colnames:
        mag_r = cat[mag_r_col]
    else:
        mag_r = flux_to_mag_LS(cat['FLUX_R'],
                               flux_mw_transmission=cat['MW_TRANSMISSION_R'])
        cat[mag_r_col] = mag_r

    if fib_mag_r_col in cat.colnames:
        fib_mag_r = cat[fib_mag_r_col]

    else:
        fib_mag_r = flux_to_mag_LS(cat['FIBERFLUX_R'],
                                   flux_mw_transmission=cat['MW_TRANSMISSION_R'])
        cat[fib_mag_r_col] = fib_mag_r

    if mag_g_col in cat.colnames:
        mag_g = cat[mag_g_col]
    else:
        mag_g = flux_to_mag_LS(cat['FLUX_G'],
                               flux_mw_transmission=cat['MW_TRANSMISSION_G'])
        cat[mag_g_col] = mag_g

    if mag_z_col in cat.colnames:
        mag_z = cat[mag_z_col]
    else:
        mag_z = flux_to_mag_LS(cat['FLUX_Z'],
                               flux_mw_transmission=cat['MW_TRANSMISSION_Z'])
        cat[mag_z_col] = mag_z

    if fibtotal_mag_col in cat.colnames:
        fibtot_mag_r = cat[fibtotal_mag_col]
    else:
        fibtot_mag_r = flux_to_mag_LS(cat['FIBERTOTFLUX_R'],
                                      flux_mw_transmission=cat['MW_TRANSMISSION_R'])
        cat[fibtotal_mag_col] = fibtot_mag_r

    if raw_mag_r_col in cat.colnames:
        raw_mag_r = cat[raw_mag_r_col]
    else:
        raw_mag_r = flux_to_mag_LS(cat['FLUX_R'],
                                   flux_mw_transmission=cat['MW_TRANSMISSION_R'])
        cat[raw_mag_r_col] = raw_mag_r

    sel = mag_r < mag_r_lim

    if 'REF_CAT' in cat.colnames:
        not_in_gaia = cat['REF_CAT'] == '  '

        sel &= ((cat['GAIA_PHOT_G_MEAN_MAG'] - raw_mag_r) > 0.6) & ~not_in_gaia
        sel |= not_in_gaia

    if if_mask:
        sel &= LS_final_masking(cat, selections='BG')

    # Fibre magnitude cut
    mask = mag_r < 17.8
    mask &= fib_mag_r < (22.9 + (mag_r - 17.8))
    mask1 = (mag_r > 17.8) & (mag_r < 20)
    mask1 &= fib_mag_r < 22.9
    sel &= (mask | mask1)

    # cut spurious objects
    mask = (mag_g-mag_r < -1) | (mag_g-mag_r > 4)
    mask |= (mag_r-mag_z < -1) | (mag_r-mag_z > 4)
    sel &= ~mask

    if all(key in cat.colnames for key in ['NOBS_G', 'NOBS_R', 'NOBS_Z']):
        mask = (cat['NOBS_G'] == 0) | (
            cat['NOBS_R'] == 0) | (cat['NOBS_Z'] == 0)
        sel &= ~mask

    mask = (mag_r > 12) & (fibtot_mag_r < 15)
    sel &= ~mask

    # Additional photometric cut
    if colour_cut_v2:
        sel &= mag_r-mag_z < (0.93*(mag_g-mag_r)-0.27)
        sel &= mag_r-mag_z > (0.4*(mag_g-mag_r)+0.07)
        sel &= mag_g - mag_r > 0.95

    else:
        sel &= mag_r-mag_z < (r_z_cut_ratio*(mag_g-mag_r)-r_z_cut)

    if all(key in cat.colnames for key in ['PARALLAX', 'PARALLAX_IVAR', 'PMRA', 'PMRA_IVAR', 'PMDEC', 'PMDEC_IVAR']):

        snr_par = np.abs(cat['PARALLAX']*np.sqrt(cat['PARALLAX_IVAR']))
        snr_pmra = np.abs(cat['PMRA']*np.sqrt(cat['PMRA_IVAR']))
        snr_pmdec = np.abs(cat['PMDEC']*np.sqrt(cat['PMDEC_IVAR']))
        mask = cat['PARALLAX'] != 0
        mask &= (snr_pmra > 3) | (snr_pmra > 3) | (snr_pmdec > 3)
        sel &= ~mask

    return sel
