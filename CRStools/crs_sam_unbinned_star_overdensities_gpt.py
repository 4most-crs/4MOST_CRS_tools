# Import necessary libraries
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from sklearn.neighbors import BallTree
from tqdm import tqdm
import numba
import pickle
import matplotlib.pyplot as plt
from matplotlib import patches

# Define a function for mask radius if needed (example: Gaia mask radius from magnitude)
def mask_radius_for_mag(mag):
    """
    Compute masking radius for a star of a given magnitude (e.g., Gaia or Tycho-2).
    Returns radius in degrees, based on a known magnitude-radius relation:contentReference[oaicite:0]{index=0}.
    """
    mag = np.asarray(mag)
    # This formula comes from Legacy Surveys (Rongpu, decam-chatter thread 12099):
    return 1630.0/3600.0 * (1.396 ** (-mag))  # 1630 arcsec at mag=0, decays as 1.396^-mag

def mask_radius_dr8(mag):
    """
    Example mask radius function used in DECam Legacy Survey (DR8) for bright star masks.
    Returns twice the radius given by mask_radius_for_mag (i.e. 'MEDIUM' mask twice the 'BRIGHT' radius:contentReference[oaicite:1]{index=1}).
    """
    return 2 * mask_radius_for_mag(mag)

# Precompile the angular offset computation using numba for performance
@numba.jit(nopython=True, parallel=True)
def compute_angular_offsets_numba(star_ra_rad, star_dec_rad, source_ra_rad, source_dec_rad, 
                                  mask_radius_arcmin, extent):
    """
    Compute angular offsets of sources from a star, normalized by the star's mask radius.
    
    Uses spherical trigonometry (great-circle distances) for accurate separation on the sky,
    and is accelerated with Numba for speed.
    
    Parameters:
    -----------
    star_ra_rad : float
        Right Ascension of the star in **radians**.
    star_dec_rad : float
        Declination of the star in **radians**.
    source_ra_rad : array(float)
        Array of R.A. positions of nearby sources in **radians**.
    source_dec_rad : array(float)
        Array of Dec. positions of nearby sources in **radians**.
    mask_radius_arcmin : float
        Mask radius for the star in **arcminutes**.
    extent : float
        Maximum extent (in units of the mask radius) to consider for offsets.
        Only offsets with |ΔRA| and |ΔDec| less than this extent will be returned.
    
    Returns:
    --------
    valid_mask : numpy.ndarray (bool)
        Boolean array indicating which sources lie within the specified extent around the star.
    x_offsets : numpy.ndarray (float)
        Offset in the RA direction (East-West), normalized by the mask radius.
    y_offsets : numpy.ndarray (float)
        Offset in the Dec direction (North-South), normalized by the mask radius.
    """
    n_sources = len(source_ra_rad)
    # Initialize output arrays
    x_offsets = np.empty(n_sources, dtype=numba.float64)
    y_offsets = np.empty(n_sources, dtype=numba.float64)
    valid_mask = np.empty(n_sources, dtype=numba.boolean)
    
    # Convert mask radius from arcminutes to radians for normalization
    mask_radius_rad = np.radians(mask_radius_arcmin / 60.0)
    
    # Precompute trigonometric terms for the star position
    cos_dec_star = np.cos(star_dec_rad)
    sin_dec_star = np.sin(star_dec_rad)
    
    for i in numba.prange(n_sources):
        # Spherical law of cosines for angular separation:
        # cos(Δθ) = sin(dec_star)*sin(dec_src) + cos(dec_star)*cos(dec_src)*cos(ΔRA)
        # Compute difference in RA:
        double_ra = source_ra_rad[i] - star_ra_rad
        # Trig values for source position
        cos_dec_src = np.cos(source_dec_rad[i])
        sin_dec_src = np.sin(source_dec_rad[i])
        cos_dRA = np.cos(double_ra)
        # Clamp cos_sep to [-1,1] to avoid numerical issues outside arccos domain
        double_cos_sep = sin_dec_star * sin_dec_src + cos_dec_star * cos_dec_src * cos_dRA
        if double_cos_sep > 1.0:
            double_cos_sep = 1.0
        elif double_cos_sep < -1.0:
            double_cos_sep = -1.0
        # Angular separation from star to source
        angular_sep = np.arccos(double_cos_sep)
        
        # Position angle of source relative to star:
        # (direction on the sky from the star to the source)
        # Use atan2 to get correct quadrant:
        # y component ~ sin(ΔRA) * cos(dec_src), x component ~ cos(dec_star)*sin(dec_src) - sin(dec_star)*cos(dec_src)*cos(ΔRA)
        double_y = np.sin(double_ra) * cos_dec_src
        double_x = cos_dec_star * sin_dec_src - sin_dec_star * cos_dec_src * cos_dRA
        position_angle = np.arctan2(double_y, double_x)
        
        # Project angular separation into X (RA) and Y (Dec) components (in radians)
        x_rad = angular_sep * np.sin(position_angle)   # East-West component of separation
        y_rad = angular_sep * np.cos(position_angle)   # North-South component of separation
        
        # Normalize offsets by the star's mask radius (in radians)
        x_norm = x_rad / mask_radius_rad
        y_norm = y_rad / mask_radius_rad
        
        # Check if this source lies within the desired extent (in units of R_mask)
        if (abs(x_norm) < extent) and (abs(y_norm) < extent):
            valid_mask[i] = True
            x_offsets[i] = x_norm
            y_offsets[i] = y_norm
        else:
            valid_mask[i] = False
            x_offsets[i] = 0.0
            y_offsets[i] = 0.0
    
    return valid_mask, x_offsets, y_offsets

class StarOverdensities:
    def __init__(self, star_catalog, galaxy_catalog,
                 star_ra_col='RA', star_dec_col='DEC', star_mag_col='G', star_mask_col='R_medium_arcsec',
                 galaxy_ra_col='RA', galaxy_dec_col='DEC',
                 mag_bins=None, mask_radius_func=None, mask_radius_units='arcsec',
                 extent=3.0, grid_bins=100, query_radius_factor=5.0):
        """
        Initialize the StarOverdensities analyzer with star and galaxy catalogs.
        
        Parameters:
        -----------
        star_catalog : pandas.DataFrame or astropy.table.Table
            Catalog of stars (e.g., Gaia stars) with at least RA, Dec, magnitude, and a mask radius column.
        galaxy_catalog : pandas.DataFrame or astropy.table.Table
            Catalog of galaxy positions (or any sources) with at least RA and Dec columns.
        star_ra_col, star_dec_col : str
            Column names for star Right Ascension and Declination (in degrees).
        star_mag_col : str
            Column name for the star magnitude (used for brightness binning).
        star_mask_col : str
            Column name for the star mask radius values in the star catalog. This typically corresponds 
            to the *medium* bright star mask radius (maskbit 11), which is twice the *bright* radius (maskbit 1):contentReference[oaicite:2]{index=2}.
        galaxy_ra_col, galaxy_dec_col : str
            Column names for RA and Dec in the galaxy/source catalog (in degrees).
        mag_bins : list of [min_mag, max_mag] pairs, optional
            Magnitude bins for splitting stars by brightness. If None, all stars are treated as one bin.
        mask_radius_func : function, optional
            A function f(magnitude) -> radius that computes the mask radius for a star of given magnitude.
            If provided, this overrides `star_mask_col` and is applied to each star's magnitude. 
            The output should be in units specified by `mask_radius_units`.
        mask_radius_units : {'deg', 'arcmin', 'arcsec'}
            Units of the mask radii provided (either via the mask radius column or the mask_radius_func output).
            This is used to convert radii to arcminutes internally for calculations.
        extent : float
            Half-size of the region (in units of mask radius) to consider around each star. 
            For example, extent=3.0 means we consider a square ±3 R_mask in size around each star.
        grid_bins : int
            Number of bins in each dimension for the 2D histogram grid (e.g., 100 -> 100x100 grid).
        query_radius_factor : float
            How far to search for galaxies around each star, as a multiple of the mask radius. 
            This should be >= extent (default 5.0 for safety) to capture all galaxies out to the edge of the plotting region.
        """
        # Convert astropy Table inputs to pandas DataFrame for consistency
        if isinstance(star_catalog, Table):
            star_catalog = star_catalog.to_pandas()
        if isinstance(galaxy_catalog, Table):
            galaxy_catalog = galaxy_catalog.to_pandas()
        
        # Store column names for later use
        self.star_ra_col = star_ra_col
        self.star_dec_col = star_dec_col
        self.star_mag_col = star_mag_col
        self.star_mask_col = star_mask_col
        self.galaxy_ra_col = galaxy_ra_col
        self.galaxy_dec_col = galaxy_dec_col
        
        # Store the catalogs as pandas DataFrames
        self.star_catalog = star_catalog.reset_index(drop=True)
        self.galaxy_catalog = galaxy_catalog.reset_index(drop=True)
        
        # Define magnitude bins for star grouping
        if mag_bins is None:
            # If no bins provided, use one bin covering the full range of star magnitudes
            all_mags = self.star_catalog[self.star_mag_col]
            mag_min = float(np.min(all_mags))
            mag_max = float(np.max(all_mags))
            mag_bins = [[mag_min, mag_max]]
        self.mag_bins = mag_bins
        # Create human-readable labels for each magnitude bin (e.g., "8 < G ≤ 13")
        self.bin_names = []
        for (low, high) in self.mag_bins:
            # Use the magnitude column name if short (like 'G'), else just 'mag'
            mag_label = self.star_mag_col if len(self.star_mag_col) <= 3 else 'mag'
            label = f"{low} < {mag_label} ≤ {high}"
            self.bin_names.append(label)
        
        # Store mask radius function and units preferences
        self.mask_radius_func = mask_radius_func
        self.mask_radius_units = mask_radius_units
        
        # Stacking and grid parameters
        self.extent = extent
        self.grid_bins = grid_bins
        self.query_radius_factor = query_radius_factor
        
        # Prepare containers for results
        self.stacks = {}     # Dictionary to hold stacked count grids for each bin
        self.bin_edges = {}  # Dictionary to hold (xedges, yedges) for each bin's grid
    
    def compute_stacks(self, save=True, stack_file='stacks.pkl', edges_file='edges.pkl'):
        """
        Compute the stacked galaxy counts around stars in each magnitude bin.
        
        For each star, this finds neighboring galaxies within a certain radius, computes their 
        offsets relative to the star (in units of the star's mask radius), and accumulates these offsets 
        into a 2D histogram. The result is a density map of galaxies around stars, stacked over all stars in the bin.
        
        Parameters:
        -----------
        save : bool
            If True, saves the resulting `stacks` and `bin_edges` dictionaries to pickle files.
        stack_file : str
            Filename for saving the stacks dictionary (only used if save=True).
        edges_file : str
            Filename for saving the bin_edges dictionary (only used if save=True).
        
        Returns:
        --------
        stacks : dict
            Dictionary mapping magnitude bin label -> 2D numpy array of shape (grid_bins, grid_bins) 
            containing the stacked galaxy counts.
        bin_edges : dict
            Dictionary mapping magnitude bin label -> (xedges, yedges) arrays for the histogram bin boundaries.
        """
        # Verify required columns exist in the input DataFrames
        for col in [self.star_ra_col, self.star_dec_col, self.star_mag_col]:
            if col not in self.star_catalog.columns:
                raise ValueError(f"Star catalog is missing required column: {col}")
        for col in [self.galaxy_ra_col, self.galaxy_dec_col]:
            if col not in self.galaxy_catalog.columns:
                raise ValueError(f"Galaxy catalog is missing required column: {col}")
        
        # Compute mask radius for each star (in arcminutes) 
        if self.mask_radius_func is not None:
            # Use the provided function on star magnitudes to get mask radii
            radii_values = self.mask_radius_func(self.star_catalog[self.star_mag_col].values)
        else:
            # Otherwise, use the mask radius column from the star catalog
            if self.star_mask_col not in self.star_catalog.columns:
                raise ValueError("No mask radius function provided and star_mask_col not in star_catalog.")
            radii_values = self.star_catalog[self.star_mask_col].values
        # Convert all mask radii to **arcminutes** for internal consistency
        if self.mask_radius_units == 'deg':
            radii_arcmin = radii_values * 60.0  # degrees -> arcminutes
        elif self.mask_radius_units == 'arcmin':
            radii_arcmin = radii_values.astype(float)
        elif self.mask_radius_units == 'arcsec':
            radii_arcmin = radii_values / 60.0   # arcseconds -> arcminutes
        else:
            raise ValueError("mask_radius_units must be one of 'deg', 'arcmin', or 'arcsec'")
        # Attach these arcminute radii to the star DataFrame for convenience
        self.star_catalog['mask_radius_arcmin'] = radii_arcmin
        
        # Build a BallTree for quick spherical neighbor searches among galaxies
        # Convert galaxy coordinates to radians and prepare (latitude, longitude) pairs as required by haversine metric
        galaxy_coords = SkyCoord(ra=self.galaxy_catalog[self.galaxy_ra_col].values * u.deg,
                                 dec=self.galaxy_catalog[self.galaxy_dec_col].values * u.deg)
        galaxy_ra_rad = galaxy_coords.ra.rad
        galaxy_dec_rad = galaxy_coords.dec.rad
        # Note: BallTree with metric='haversine' expects input as [dec (lat), ra (lon)] in radians:contentReference[oaicite:3]{index=3}
        galaxy_points = np.vstack([galaxy_dec_rad, galaxy_ra_rad]).T
        tree = BallTree(galaxy_points, metric='haversine')
        
        # Set up the grid for stacking (same for all bins): from -extent to +extent in normalized units
        nbins = self.grid_bins
        xedges = np.linspace(-self.extent, self.extent, nbins + 1)
        yedges = np.linspace(-self.extent, self.extent, nbins + 1)
        
        # Clear any previous results
        self.stacks.clear()
        self.bin_edges.clear()
        
        # Loop through each magnitude bin and stack galaxy counts
        for bin_idx, bin_label in enumerate(self.bin_names):
            low_mag, high_mag = self.mag_bins[bin_idx]
            # Select stars in this bin: (low < mag <= high)
            star_mask = (self.star_catalog[self.star_mag_col].values > low_mag) & \
                        (self.star_catalog[self.star_mag_col].values <= high_mag)
            stars_bin = self.star_catalog[star_mask]
            num_stars = len(stars_bin)
            if num_stars == 0:
                print(f"No stars in bin {bin_label}, skipping.")
                continue
            
            # Initialize an empty count grid for this bin
            count_grid = np.zeros((nbins, nbins), dtype=np.float64)
            
            # Precompute star coordinates in radians for this bin
            star_coords = SkyCoord(ra=stars_bin[self.star_ra_col].values * u.deg,
                                   dec=stars_bin[self.star_dec_col].values * u.deg)
            star_ra_rad = star_coords.ra.rad
            star_dec_rad = star_coords.dec.rad
            star_mask_radii = stars_bin['mask_radius_arcmin'].values  # mask radius (arcmin) for each star in this bin
            
            # Iterate over each star and accumulate galaxy counts
            for i in tqdm(range(num_stars), desc=f"Processing {bin_label}", leave=False):
                ra_rad = star_ra_rad[i]
                dec_rad = star_dec_rad[i]
                mask_rad_arcmin = star_mask_radii[i]
                
                # Determine search radius for this star = mask_radius * query_radius_factor (convert to radians for BallTree)
                query_radius_rad = np.radians((mask_rad_arcmin * self.query_radius_factor) / 60.0)
                # Find all galaxies within this radius using the BallTree (returns indices of neighbors)
                star_point = np.array([[star_dec_rad[i], star_ra_rad[i]]])  # star location in (dec, ra) radians
                neighbor_idx = tree.query_radius(star_point, r=query_radius_rad)[0]
                if len(neighbor_idx) == 0:
                    continue  # no galaxies near this star within the search radius
                
                # Get the neighbor galaxies' coordinates in radians
                neighbor_ra = galaxy_ra_rad[neighbor_idx]
                neighbor_dec = galaxy_dec_rad[neighbor_idx]
                # Compute normalized offsets for all neighbor galaxies relative to the star
                valid_mask, x_off, y_off = compute_angular_offsets_numba(ra_rad, dec_rad,
                                                                         neighbor_ra, neighbor_dec,
                                                                         mask_rad_arcmin, self.extent)
                # Filter to only those neighbors within the stacking extent (valid_mask)
                if not np.any(valid_mask):
                    continue
                x_off = x_off[valid_mask]
                y_off = y_off[valid_mask]
                # Bin the offsets into our 2D grid:
                x_bin_idx = np.digitize(x_off, xedges) - 1  # digitize gives 1-based bin indices
                y_bin_idx = np.digitize(y_off, yedges) - 1
                # Accumulate counts in the grid
                for xb, yb in zip(x_bin_idx, y_bin_idx):
                    if 0 <= xb < nbins and 0 <= yb < nbins:
                        count_grid[yb, xb] += 1
            # Store the stacked counts and edges for this bin
            self.stacks[bin_label] = count_grid
            self.bin_edges[bin_label] = (xedges, yedges)
            print(f"Completed bin {bin_label}: stacked {int(count_grid.sum())} galaxy counts.")
        
        # Optionally save results to disk
        if save:
            with open(stack_file, 'wb') as f:
                pickle.dump(self.stacks, f)
            with open(edges_file, 'wb') as f:
                pickle.dump(self.bin_edges, f)
            print(f"Saved stack results to files: {stack_file}, {edges_file}")
        
        return self.stacks, self.bin_edges
    
    def save_results(self, stack_file='stacks.pkl', edges_file='edges.pkl'):
        """
        Save the current `stacks` and `bin_edges` results to pickle files.
        """
        if not self.stacks:
            raise RuntimeError("No results to save. Run compute_stacks() first.")
        with open(stack_file, 'wb') as f:
            pickle.dump(self.stacks, f)
        with open(edges_file, 'wb') as f:
            pickle.dump(self.bin_edges, f)
        print(f"Results saved to {stack_file} and {edges_file}")
    
    def load_results(self, stack_file='stacks.pkl', edges_file='edges.pkl'):
        """
        Load previously computed `stacks` and `bin_edges` from pickle files.
        """
        with open(stack_file, 'rb') as f:
            self.stacks = pickle.load(f)
        with open(edges_file, 'rb') as f:
            self.bin_edges = pickle.load(f)
        # Update bin_names in case they were not set (keys are bin labels)
        self.bin_names = list(self.stacks.keys())
        print(f"Results loaded from {stack_file} and {edges_file}")
    
    def plot_stacks(self, axes=None, cmap='seismic', log_scale=True, save_file='overdensity_stacks.pdf'):
        """
        Plot the stacked galaxy overdensity maps for each magnitude bin.
        
        Creates a heatmap for each bin showing the density of galaxies around stars, 
        in units of the stars' mask radius. By default, it plots log10 of the density ratio (ρ/⟨ρ⟩) 
        so that 0 indicates average background density.
        
        Parameters:
        -----------
        axes : matplotlib Axes or array-like of Axes, optional
            If provided, draw the plots on these Axes. The number of Axes should match the number of bins.
            If None, a new Figure with an appropriate grid of subplots is created.
        cmap : str
            Colormap for the heatmaps. Default 'seismic' (blue-white-red diverging).
        log_scale : bool
            If True, color scale represents log10(ρ/⟨ρ⟩) (logarithmic density contrast). If False, uses linear contrast (ρ/⟨ρ⟩ − 1).
        save_file : str or None
            If set to a filename (e.g. 'output.pdf'), the figure is saved to that file. If None, the figure is not saved.
        """
        if not self.stacks:
            raise RuntimeError("No stack data to plot. Run compute_stacks() first or load_results().")
        
        bin_labels = list(self.stacks.keys())
        n_bins = len(bin_labels)
        
        # Determine subplot arrangement if no axes provided
        external_axes = False
        if axes is None:
            # Create a new figure with subplots for each bin
            if n_bins == 1:
                fig, axes_arr = plt.subplots(1, 1, figsize=(6, 5))
                axes_arr = np.array([axes_arr])
            elif n_bins <= 2:
                fig, axes_arr = plt.subplots(1, n_bins, figsize=(6*n_bins, 5))
            elif n_bins <= 4:
                fig, axes_arr = plt.subplots(2, 2, figsize=(12, 10))
            else:
                # For more bins, use 3 columns and enough rows
                n_cols = 3
                n_rows = int(np.ceil(n_bins / n_cols))
                fig, axes_arr = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            axes_list = np.array(axes_arr).reshape(-1)  # flatten to list
        else:
            # Use provided axes
            external_axes = True
            if isinstance(axes, plt.Axes):
                axes_list = [axes]
            else:
                axes_list = list(np.array(axes).flat)
            if len(axes_list) < n_bins:
                raise ValueError(f"Not enough axes provided for {n_bins} bins.")
            fig = plt.gcf()
        
        # Plot each magnitude bin's heatmap
        for i, bin_label in enumerate(bin_labels):
            if i >= len(axes_list):
                break  # safety check
            ax = axes_list[i]
            data = self.stacks[bin_label]
            xedges, yedges = self.bin_edges[bin_label]
            # Calculate density relative to mean
            if np.count_nonzero(data) == 0:
                mean_density = 0.0
            else:
                mean_density = np.mean(data[data > 0])
            if mean_density <= 0:
                mean_density = 1.0  # avoid zero or negative mean
            if log_scale:
                # Logarithmic density contrast: log10(ρ / ⟨ρ⟩)
                density_ratio = data / mean_density
                with np.errstate(divide='ignore'):
                    log_ratio = np.log10(density_ratio)  # log10 of ratio
                # Replace -inf (from zero counts) with lowest finite value for plotting
                if np.isneginf(log_ratio).any():
                    min_finite = np.min(log_ratio[np.isfinite(log_ratio)])
                    log_ratio[np.isneginf(log_ratio)] = min_finite
                plot_data = log_ratio
            else:
                # Linear density contrast: (ρ/⟨ρ⟩) - 1, so 0 means average density
                density_ratio = data / mean_density
                plot_data = density_ratio - 1.0
            
            # Set symmetric color scale around 0 (so blue = underdense, red = overdense)
            if log_scale:
                max_abs = np.nanmax(np.abs(plot_data))
                vmin, vmax = -max_abs, max_abs if max_abs > 0 else (-1.0, 1.0)
            else:
                max_abs = np.nanmax(np.abs(plot_data))
                vmin, vmax = -max_abs, max_abs
            
            # Plot the 2D density map
            pcm = ax.pcolormesh(xedges, yedges, plot_data, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
            # Draw a circle of radius = 1 (one mask radius) for reference
            circle = patches.Circle((0, 0), 1.0, fill=False, color='black', linewidth=1.5)
            ax.add_patch(circle)
            # Set plot labels and title
            ax.set_aspect('equal')
            ax.set_xlabel(r'$\Delta \mathrm{RA} \,/\, R_{\mathrm{mask}}$')
            ax.set_ylabel(r'$\Delta \mathrm{Dec} \,/\, R_{\mathrm{mask}}$')
            ax.set_title(bin_label)
            ax.grid(alpha=0.3, linestyle=':')
            # Add a colorbar if we created the figure (to avoid duplicating colorbars when axes are given)
            if not external_axes:
                cbar = plt.colorbar(pcm, ax=ax)
                if log_scale:
                    cbar.set_label(r'$\log_{10}(\rho/\bar{\rho})$')  # log10 density contrast
                else:
                    cbar.set_label(r'$\frac{\rho - \bar{\rho}}{\bar{\rho}}$')  # linear density contrast
        
        # Hide any unused subplots (in case the grid is larger than n_bins)
        if not external_axes:
            for j in range(len(bin_labels), len(axes_list)):
                axes_list[j].set_visible(False)
            plt.tight_layout()
        # Save to file if requested
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')
            print(f"Plot saved to {save_file}")
        # Display the plot if we created new figure/axes
        if not external_axes:
            plt.show()

if __name__ == "__main__":


    Legacy_BG_path = "/its/home/bb345/5-4most_data/CRS/target_catalogues/BG/full_legacy_no_colour_sel/.archive/reduced/desi_bg_nomaskbit_mask_4M_reduced_columns.fits"
    Gaia_path = "/its/home/bb345/5-4most_data/other_data/gaia_sources/gaia-mask-dr10_bg_foot.fits"
    
    BG = Table.read(Legacy_BG_path)
    BG = BG.to_pandas()
    BG = BG[BG.isBG_4M_v2 & BG.in_S8]

    gaia = Table.read(Gaia_path)
    gaia = gaia.to_pandas()

    mask_radius_col = 'R_bright_arcsec'

    # Initialize the analyzer; specify appropriate columns and units
    analyzer = StarOverdensities(star_catalog=gaia, galaxy_catalog=BG,
                                star_ra_col='RA', star_dec_col='DEC', 
                                star_mag_col='G', star_mask_col='R_medium_arcsec',
                                galaxy_ra_col='RA', galaxy_dec_col='DEC',
                                mask_radius_units='arcsec',
                                mag_bins=[[0,8],[8,13],[13,16]])  # example magnitude bins

    # Compute the stacks (this will also save stacks.pkl and edges.pkl by default)
    analyzer.compute_stacks()

    # Plot the results (this will create a PDF file by default)
    analyzer.plot_stacks()


