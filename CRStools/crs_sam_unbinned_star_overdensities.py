import numba
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.neighbors import BallTree
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle


size_ratio = 1
sub_width, sub_height = size_ratio*10/3, size_ratio*2.8
SMALL_SIZE = 7
MEDIUM_SIZE = 9
BIGGER_SIZE = 10


rc_default = {}
rc_default['font.family'] = 'serif'
rc_default['font.size'] = SMALL_SIZE
rc_default['axes.labelsize'] = MEDIUM_SIZE
rc_default['axes.labelweight'] = 'normal'
rc_default['axes.linewidth'] = 1.0
rc_default['axes.titlesize'] = MEDIUM_SIZE
rc_default['xtick.labelsize'] = SMALL_SIZE
rc_default['ytick.labelsize'] = SMALL_SIZE
rc_default['legend.fontsize'] = SMALL_SIZE
rc_default['figure.titlesize'] = BIGGER_SIZE
rc_default['lines.linewidth'] = 1
rc_default['lines.markersize'] = 4
rc_default['figure.figsize'] = (sub_width, sub_height)
rc_default['savefig.dpi'] = 400

# Latex related
rc_default['text.usetex'] = True
rc_default['mathtext.fontset'] = 'custom'
rc_default['mathtext.rm'] = 'Bitstream Vera Sans'
rc_default['mathtext.it'] = 'Bitstream Vera Sans:italic'
rc_default['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rcParams.update(rc_default)
plt.style.use('tableau-colorblind10')


# It is good practice to keep the Numba function separate from the class
# if it doesn't need access to the class's internal state (self).
@numba.jit(nopython=True, parallel=True)
def compute_angular_offsets_numba(star_ra_rad, star_dec_rad, source_ra_rad, source_dec_rad,
                                  mask_radius_arcmin, extent):
    """
    Numba-accelerated function to compute angular offsets in units of mask radius.
    Uses proper spherical trigonometry without the flat-sky approximation.

    Parameters:
    -----------
    star_ra_rad, star_dec_rad : float
        Star position in radians.
    source_ra_rad, source_dec_rad : np.ndarray
        Source positions in radians.
    mask_radius_arcmin : float
        Mask radius in arcminutes.
    extent : float
        Maximum extent in units of mask radius to consider a source valid.

    Returns:
    --------
    valid_mask : np.ndarray of bool
        Mask for sources within the specified extent.
    x_offsets, y_offsets : np.ndarray
        Normalised offsets in units of mask radius for valid sources.
    """
    n_sources = len(source_ra_rad)
    x_offsets = np.empty(n_sources, dtype=numba.float64)
    y_offsets = np.empty(n_sources, dtype=numba.float64)
    valid_mask = np.empty(n_sources, dtype=numba.boolean)

    # Convert mask radius to radians for trigonometric calculations
    mask_radius_rad = np.radians(mask_radius_arcmin / 60.0)

    # Precompute trigonometric values for the star's position
    cos_dec_star = np.cos(star_dec_rad)
    sin_dec_star = np.sin(star_dec_rad)

    for i in numba.prange(n_sources):
        # Angular separation calculation using the Haversine formula
        dra = source_ra_rad[i] - star_ra_rad
        cos_dec_source = np.cos(source_dec_rad[i])
        sin_dec_source = np.sin(source_dec_rad[i])
        cos_dra = np.cos(dra)
        
        # Clamp to avoid numerical errors from floating-point inaccuracies
        cos_sep = max(-1.0, min(1.0, sin_dec_star * sin_dec_source + cos_dec_star * cos_dec_source * cos_dra))
        angular_sep = np.arccos(cos_sep)

        # Position angle calculation
        y_pa = np.sin(dra) * cos_dec_source
        x_pa = cos_dec_star * sin_dec_source - sin_dec_star * cos_dec_source * cos_dra
        position_angle = np.arctan2(y_pa, x_pa)

        # Convert separation and position angle to Cartesian offsets in radians
        x_rad = angular_sep * np.sin(position_angle)
        y_rad = angular_sep * np.cos(position_angle)

        # Normalise offsets by the mask radius
        x_norm = x_rad / mask_radius_rad
        y_norm = y_rad / mask_radius_rad

        # Check if the source is within the defined extent
        if abs(x_norm) < extent and abs(y_norm) < extent:
            valid_mask[i] = True
            x_offsets[i] = x_norm
            y_offsets[i] = y_norm
        else:
            valid_mask[i] = False
            # Assign a value even if not valid, though it will be filtered out
            x_offsets[i] = 0.0
            y_offsets[i] = 0.0
            
    return valid_mask, x_offsets, y_offsets


class StarOverdensityAnalyser:
    """
    A class to analyse the overdensity of galaxies around stars.

    This class takes star and galaxy catalogues as pandas DataFrames and
    computes stacked 2D histograms of galaxy positions relative to star
    positions, normalised by a mask radius. It is designed to be generic
    and configurable.
    """

    def __init__(self,
                 stars_df: pd.DataFrame,
                 galaxies_df: pd.DataFrame,
                 star_ra_col: str = 'RA',
                 star_dec_col: str = 'DEC',
                 star_mag_col: str = 'G',
                 star_mask_radius_col: str = None,
                 galaxy_ra_col: str = 'RA',
                 galaxy_dec_col: str = 'DEC',
                 mask_radius_func=None,
                 mask_radius_unit: str = 'arcsec',
                 magnitude_bins: list = [[0, 8], [8, 13], [13, 16]]):
        """
        Initialises the StarOverdensityAnalyser.

        Parameters:
        -----------
        stars_df : pd.DataFrame
            DataFrame containing the star catalogue.
        galaxies_df : pd.DataFrame
            DataFrame containing the galaxy catalogue.
        star_ra_col, star_dec_col, star_mag_col : str, optional
            Column names for star RA, Dec, and magnitude. Defaults are 'RA', 'DEC', 'G'.
        star_mask_radius_col : str, optional
            Column name for pre-calculated mask radii in the stars DataFrame. If None,
            `mask_radius_func` must be provided. Default is None.
        galaxy_ra_col, galaxy_dec_col : str, optional
            Column names for galaxy RA and Dec. Defaults are 'RA', 'DEC'.
        mask_radius_func : callable, optional
            A function that takes a magnitude array and returns a mask radius array.
            Required if `star_mask_radius_col` is not provided.
        mask_radius_unit : str, optional
            The unit of the mask radius ('degree', 'arcmin', 'arcsec'). Default is 'arcsec'.
        magnitude_bins : list of lists, optional
            A list of [min, max] pairs defining the magnitude bins for stacking.
        """
        self.stars_df = stars_df.copy()
        self.galaxies_df = galaxies_df.copy()
        self.star_ra_col = star_ra_col
        self.star_dec_col = star_dec_col
        self.star_mag_col = star_mag_col
        self.galaxy_ra_col = galaxy_ra_col
        self.galaxy_dec_col = galaxy_dec_col
        self.magnitude_bins = magnitude_bins
        self.bin_names = [f"{b[0]} < G < {b[1]}" for b in self.magnitude_bins]
        
        self.extent = 3  # Stacking extent in units of mask radius
        self.stacks = {}
        self.bin_edges = {}

        # Validate and process mask radius
        if star_mask_radius_col is None and mask_radius_func is None:
            raise ValueError("Either `star_mask_radius_col` or `mask_radius_func` must be provided.")
        
        if star_mask_radius_col:
            radius_values = self.stars_df[star_mask_radius_col].values
        else:
            radius_values = mask_radius_func(self.stars_df[self.star_mag_col].values)

        # Convert mask radius to a consistent unit (arcminutes)
        if mask_radius_unit == 'arcsec':
            self.stars_df['mask_radius_arcmin'] = radius_values / 60.0
        elif mask_radius_unit == 'arcmin':
            self.stars_df['mask_radius_arcmin'] = radius_values
        elif mask_radius_unit == 'degree':
            self.stars_df['mask_radius_arcmin'] = radius_values * 60.0
        else:
            raise ValueError("`mask_radius_unit` must be 'degree', 'arcmin', or 'arcsec'.")

    def get_stacks(self, nbins: int = 100):
        """
        Computes the stacked overdensity maps for each magnitude bin.

        This method builds a BallTree for efficient spatial queries on the galaxy
        catalogue. It then iterates through each star, finds nearby galaxies,
        computes their normalised positions, and stacks them into a 2D histogram.

        Parameters:
        -----------
        nbins : int, optional
            The number of bins for the 2D histogram stack. Default is 100.
        """
        print("Preparing galaxy catalogue for spatial queries...")
        # Prepare galaxy coordinates for BallTree
        galaxy_coords = SkyCoord(
            ra=self.galaxies_df[self.galaxy_ra_col].values * u.deg,
            dec=self.galaxies_df[self.galaxy_dec_col].values * u.deg
        )
        galaxy_ra_rad = galaxy_coords.ra.rad
        galaxy_dec_rad = galaxy_coords.dec.rad
        galaxy_coords_sphere = np.column_stack([galaxy_dec_rad, galaxy_ra_rad])
        
        print("Building BallTree for galaxies...")
        tree = BallTree(galaxy_coords_sphere, metric='haversine')
        
        # Query radius needs to be larger than the stacking extent to catch all sources
        query_extent_factor = self.extent + 1

        for i, gbin in enumerate(self.magnitude_bins):
            bin_name = self.bin_names[i]
            print(f"\nProcessing magnitude bin: {bin_name}")
            
            # Select stars in the current magnitude bin
            sel = ((self.stars_df[self.star_mag_col] > gbin[0]) &
                   (self.stars_df[self.star_mag_col] <= gbin[1]))
            stars_in_bin = self.stars_df[sel]

            if len(stars_in_bin) == 0:
                print(f"No stars found in bin {bin_name}.")
                continue

            # Prepare star coordinates for the current bin
            star_coords = SkyCoord(
                ra=stars_in_bin[self.star_ra_col].values * u.deg,
                dec=stars_in_bin[self.star_dec_col].values * u.deg
            )
            radii_arcmin = stars_in_bin['mask_radius_arcmin'].values

            # Initialise stack and bin edges for this magnitude bin
            xedges = np.linspace(-self.extent, self.extent, nbins + 1)
            yedges = np.linspace(-self.extent, self.extent, nbins + 1)
            self.bin_edges[bin_name] = (xedges, yedges)
            self.stacks[bin_name] = np.zeros((nbins, nbins), dtype=np.float64)

            # Process each star in the bin
            for j in tqdm(range(len(stars_in_bin)), desc=f"Stacking {bin_name}"):
                star_ra_rad = star_coords.ra.rad[j]
                star_dec_rad = star_coords.dec.rad[j]
                mask_radius_arcmin = radii_arcmin[j]

                # Define a query radius in radians for the BallTree search
                query_radius_rad = np.radians(mask_radius_arcmin * query_extent_factor / 60.0)
                star_coord_sphere = np.array([[star_dec_rad, star_ra_rad]])
                
                # Find indices of galaxies within the query radius
                indices = tree.query_radius(star_coord_sphere, r=query_radius_rad)[0]
                if len(indices) == 0:
                    continue

                # Get coordinates of nearby galaxies
                nearby_galaxy_ra = galaxy_ra_rad[indices]
                nearby_galaxy_dec = galaxy_dec_rad[indices]
                
                # Compute normalised offsets using the Numba-accelerated function
                valid_mask, x_offsets, y_offsets = compute_angular_offsets_numba(
                    star_ra_rad, star_dec_rad,
                    nearby_galaxy_ra, nearby_galaxy_dec,
                    mask_radius_arcmin, self.extent
                )

                # Add valid offsets to the 2D histogram
                x_valid = x_offsets[valid_mask]
                y_valid = y_offsets[valid_mask]
                
                if len(x_valid) > 0:
                    H, _, _ = np.histogram2d(y_valid, x_valid, bins=(yedges, xedges))
                    self.stacks[bin_name] += H
            
            print(f"Completed bin {bin_name}: total counts = {self.stacks[bin_name].sum()}")

    def save_results(self, stacks_filepath: str = 'stacks.pkl', edges_filepath: str = 'edges.pkl'):
        """Saves the computed stacks and bin edges to pickle files."""
        with open(stacks_filepath, 'wb') as f:
            pickle.dump(self.stacks, f)
        with open(edges_filepath, 'wb') as f:
            pickle.dump(self.bin_edges, f)
        print(f"Results saved to {stacks_filepath} and {edges_filepath}")

    def load_results(self, stacks_filepath: str = 'stacks.pkl', edges_filepath: str = 'edges.pkl'):
        """Loads computed stacks and bin edges from pickle files."""
        with open(stacks_filepath, 'rb') as f:
            self.stacks = pickle.load(f)
        with open(edges_filepath, 'rb') as f:
            self.bin_edges = pickle.load(f)
        print(f"Results loaded from {stacks_filepath} and {edges_filepath}")

    def plot_stacks(self, output_filename: str = 'overdensity_stacks.png'):
        """
        Plots the stacked overdensity maps for each magnitude bin.
        
        Parameters:
        -----------
        output_filename : str, optional
            The filename for the saved plot. Default is 'overdensity_stacks.png'.
        """
        if not self.stacks:
            print("No stacks to plot! Run get_stacks() first.")
            return

        n_bins = len(self.stacks)
        if n_bins == 0:
            print("No data in stacks.")
            return

        # Determine the subplot layout
        if n_bins <= 3:
            n_rows, n_cols = 1, n_bins
        else:
            n_cols = 3
            n_rows = (n_bins + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(sub_width * 2, sub_height), squeeze=False)
        axes = axes.flatten()

        for i, (bin_name, stacked_map) in enumerate(self.stacks.items()):
            ax = axes[i]
            xedges, yedges = self.bin_edges[bin_name]
            
            # Normalise by the mean density outside the central region to get overdensity
            # Here we calculate mean density from an annulus between 1.5 and `extent` radii
            x_cent, y_cent = (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2
            xx, yy = np.meshgrid(x_cent, y_cent)
            r = np.sqrt(xx**2 + yy**2)
            
            background_mask = (r > 1.5) & (r < self.extent)
            mean_density = np.mean(stacked_map[background_mask])

            if mean_density > 0:
                # Use log10 for better visualisation of over/under densities
                log_overdensity = np.log10(stacked_map / mean_density)
                # Handle -inf where stacked_map is 0
                log_overdensity[np.isneginf(log_overdensity)] = np.nan
                vmax = np.nanpercentile(np.abs(log_overdensity), 98) # Use 98th percentile to avoid extreme outliers
                vmin = -vmax
            else:
                log_overdensity = np.zeros_like(stacked_map)
                vmin, vmax = -1, 1

            pcm = ax.pcolormesh(xedges, yedges, log_overdensity.T, cmap='seismic', vmin=vmin, vmax=vmax, shading='auto')
            
            ax.set_xlabel(r'$\Delta \mathrm{RA} / R_{\mathrm{mask}}$')
            ax.set_ylabel(r'$\Delta \mathrm{Dec} / R_{\mathrm{mask}}$')
            ax.set_title(f'Galaxy Density around Stars ({bin_name})')
            ax.set_aspect('equal')
            ax.set_xlim(-self.extent, self.extent)
            ax.set_ylim(-self.extent, self.extent)
            
            # Add a circle representing the mask radius
            circle = patches.Circle((0, 0), 1, fill=False, color='black', linestyle='--', linewidth=2)
            ax.add_patch(circle)
            circle = patches.Circle((0, 0), 2, fill=False, color='blue', linestyle=':', linewidth=2)
            ax.add_patch(circle)

            cbar = plt.colorbar(pcm, ax=ax)
            cbar.set_label(r'$\log_{10}(\rho / \bar{\rho})$')
            ax.grid(True, linestyle=':', alpha=0.5)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.show()

# --- Example Usage ---
if __name__ == '__main__':
    
    # 1. Define a custom mask radius function for Gaia stars (example)
    def mask_radius_gaia(g_mag):
        """
        Example mask radius function. Returns radius in arcseconds.
        This is based on the function `mask_radius_for_mag` from the original code.
        """
        g_mag = np.asarray(g_mag)
        # The formula returns radius in degrees, so we convert to arcseconds.
        return 0.5*(1630.0 * 1.396**(-g_mag))

    Legacy_BG_path = "/its/home/bb345/5-4most_data/CRS/target_catalogues/BG/full_legacy_no_colour_sel/.archive/reduced/desi_bg_nomaskbit_mask_4M_reduced_columns.fits"
    Gaia_path = "/its/home/bb345/5-4most_data/other_data/gaia_sources/gaia-mask-dr10_bg_foot.fits"
    
    BG = Table.read(Legacy_BG_path)
    BG = BG.to_pandas()
    BG = BG[BG.isBG_4M_v2 & BG.in_S8]

    gaia = Table.read(Gaia_path)
    gaia = gaia.to_pandas()

    ra_min, ra_max = 25, 85
    dec_min, dec_max = -60, -40



    # # 2. Create dummy data using Astropy Tables, as requested
    # print("Creating dummy Star and Galaxy catalogues...")
    # # Dummy Star Catalogue
    # n_stars = 1000
    # star_data = Table({
    #     'RA': np.random.uniform(130, 140, n_stars),
    #     'DEC': np.random.uniform(-5, 5, n_stars),
    #     'G': np.random.uniform(6, 16, n_stars)
    # })
    # # Add pre-calculated radii columns for CRS-like catalogue
    # star_data['R_medium_arcsec'] = mask_radius_gaia(star_data['G'])
    # star_data['R_bright_arcsec'] = star_data['R_medium_arcsec'] / 2.0

    # # Dummy Galaxy Catalogue
    # n_galaxies = 20000
    # galaxy_data = Table({
    #     'RA': np.random.uniform(130, 140, n_galaxies),
    #     'DEC': np.random.uniform(-5, 5, n_galaxies)
    # })

    # Convert Astropy Tables to pandas DataFrames
    # stars_df = star_data.to_pandas()
    # galaxies_df = galaxy_data.to_pandas()

    stars_df = gaia#[(gaia['RA'] > ra_min) & (gaia['RA'] < ra_max) & (gaia['DEC'] > dec_min) & (gaia['DEC'] < dec_max)]
    galaxies_df = BG#[(BG['RA'] > ra_min) & (BG['RA'] < ra_max) & (BG['DEC'] > dec_min) & (BG['DEC'] < dec_max)]

    print(f"Generated {len(stars_df)} stars and {len(galaxies_df)} galaxies.")

    # --- Scenario 1: Using a pre-calculated radius column ---
    print("\n--- Running Scenario 1: Using a pre-calculated radius column ---")
    
    # # In this scenario, we pretend our catalogue has a column 'R_medium_arcsec'
    # # that we want to use for the mask radius.
    # analyser_scenario_1 = StarOverdensityAnalyser(
    #     stars_df=stars_df,
    #     galaxies_df=galaxies_df,
    #     star_ra_col='RA',
    #     star_dec_col='DEC',
    #     star_mag_col='G',
    #     star_mask_radius_col='R_medium_arcsec', # Specify the column
    #     mask_radius_unit='arcsec', # Specify its unit
    #     magnitude_bins=[[8, 10], [10, 12]] # Different bins for variety
    # )

    # # Compute and plot the stacks
    # analyser_scenario_1.get_stacks(nbins=50)
    # analyser_scenario_1.plot_stacks(output_filename='overdensity_scenario_1.png')
    
    # --- Scenario 2: Using a custom function to calculate radius ---
    print("\n--- Running Scenario 2: Using a custom function for radius ---")

    # In this scenario, we provide a function to calculate the radius on-the-fly
    # based on the magnitude.
    analyser_scenario_2 = StarOverdensityAnalyser(
        stars_df=stars_df,
        galaxies_df=galaxies_df,
        star_ra_col='RA',
        star_dec_col='DEC',
        star_mag_col='G',
        mask_radius_func=mask_radius_gaia, # Provide the function
        mask_radius_unit='arcsec', # Specify the unit the function returns
        magnitude_bins=[[8, 11], [11, 13]] # Different bins for variety
    )
    
    # Compute and plot the stacks
    analyser_scenario_2.get_stacks(nbins=200)
    analyser_scenario_2.plot_stacks(output_filename='overdensity_scenario_2.png')

    print("\nAnalysis complete. Check for 'overdensity_scenario_1.png' and 'overdensity_scenario_2.png'.")