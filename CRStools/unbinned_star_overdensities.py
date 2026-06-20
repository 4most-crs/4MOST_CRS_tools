import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numba
from sklearn.neighbors import BallTree
import pickle



def mask_radius_waves(g):
    g = np.asarray(g)
    r = np.zeros_like(g, dtype=float)
    mask1 = (g > 3.5) & (g < 16)
    mask2 = g <= 3.5
    r[mask1] = (10 ** (1.3 - 0.13 * g[mask1]))
    r[mask2] = 7
    return r

@numba.jit(nopython=True, parallel=True)
def compute_angular_offsets_numba(star_ra_rad, star_dec_rad, source_ra_rad, source_dec_rad, 
                                  mask_radius_arcmin, extent):
    """
    Numba-accelerated function to compute angular offsets in units of mask radius.
    Uses proper spherical trigonometry without flat-sky approximation.
    
    Parameters:
    -----------
    star_ra_rad, star_dec_rad : float
        Star position in radians
    source_ra_rad, source_dec_rad : array
        Source positions in radians
    mask_radius_arcmin : float
        Mask radius in arcminutes
    extent : float
        Maximum extent in units of mask radius
    
    Returns:
    --------
    valid_mask : array of bool
        Mask for sources within extent
    x_offsets, y_offsets : arrays
        Normalized offsets in units of mask radius
    """
    n_sources = len(source_ra_rad)
    x_offsets = np.empty(n_sources, dtype=numba.float64)
    y_offsets = np.empty(n_sources, dtype=numba.float64)
    valid_mask = np.empty(n_sources, dtype=numba.boolean)
    
    # Convert mask radius to radians
    mask_radius_rad = np.radians(mask_radius_arcmin / 60.0)
    
    # Precompute star position trigonometry
    cos_dec_star = np.cos(star_dec_rad)
    sin_dec_star = np.sin(star_dec_rad)
    
    for i in numba.prange(n_sources):
        # Compute angular separation using spherical trigonometry
        dra = source_ra_rad[i] - star_ra_rad
        cos_dec_source = np.cos(source_dec_rad[i])
        sin_dec_source = np.sin(source_dec_rad[i])
        
        # Haversine formula components
        cos_dra = np.cos(dra)
        sin_dra = np.sin(dra)
        
        # Angular separation
        cos_sep = sin_dec_star * sin_dec_source + cos_dec_star * cos_dec_source * cos_dra
        
        # Clamp to valid range to avoid numerical issues
        cos_sep = max(-1.0, min(1.0, cos_sep))
        angular_sep = np.arccos(cos_sep)
        
        # Position angle (bearing from star to source)
        # Using atan2 for proper quadrant handling
        y_pa = sin_dra * cos_dec_source
        x_pa = cos_dec_star * sin_dec_source - sin_dec_star * cos_dec_source * cos_dra
        position_angle = np.arctan2(y_pa, x_pa)
        
        # Convert to Cartesian offsets in radians
        x_rad = angular_sep * np.sin(position_angle)
        y_rad = angular_sep * np.cos(position_angle)
        
        # Normalize by mask radius
        x_norm = x_rad / mask_radius_rad
        y_norm = y_rad / mask_radius_rad
        
        # Check if within extent
        if abs(x_norm) < extent and abs(y_norm) < extent:
            valid_mask[i] = True
            x_offsets[i] = x_norm
            y_offsets[i] = y_norm
        else:
            valid_mask[i] = False
            x_offsets[i] = 0.0
            y_offsets[i] = 0.0
    
    return valid_mask, x_offsets, y_offsets


class star_overdensites:
    def __init__(self,
                 waves_filepath='/its/home/bb345/5-4most_data/WAVES/target_catalogues/reduced_catalogues/raw_parent_wide_1p2/WAVES-S_1p2_ultra_light.parquet',
                 gaiastar_filepath='/its/home/bb345/5-4most_data/WAVES/masks/gaiastarmaskwaves.csv',
                 waves_region='S',
                 filters = ['Ghosts and Artefacts', 'Ghosts and no Artefacts', 'No Ghosts or Artefacts'],
                 gaia_g_bins=[[0, 8], [8, 13], [13, 15], [15, 16]]):

        self.waves_filepath = waves_filepath
        self.gaiastar_filepath = gaiastar_filepath
        self.waves_region = waves_region

        if self.waves_region not in ['N', 'S']:
            raise ValueError('waves_region must be either N or S')

        self.gaia_bins = gaia_g_bins
        self.bin_names = [f"{b[0]}<G<{b[1]}" for b in self.gaia_bins]
        print(f'bin names: {self.bin_names}')

        self.filters = filters
        
        # Define filter conditions
        self.filter_conditions = {
            'Ghosts and Artefacts': lambda cat: pd.Series([True] * len(cat), index=cat.index),  # No filtering - all data
            'Ghosts and no Artefacts': lambda cat: (cat['class'] != 'artefact'),  # Remove artefacts only
            'No Ghosts or Artefacts': lambda cat: (cat['ghostmask'] == 0) & (cat['class'] != 'artefact')  # Remove both
        }

        self.stacks = {}
        self.bin_edges = {}  # Holds xedges, yedges for each bin and filter
        self.nside = 1024 * 32
        self.extent = 3  # R units


    def load_waves(self):
        self.cat = pq.read_table(self.waves_filepath, columns=['RAmax', 'Decmax', 'class', 'ghostmask', 'duplicate']).to_pandas()
        self.cat = self.cat[(self.cat['duplicate'] == 0)].reset_index(drop=True)

    def load_gaia_stars(self):
        self.all_stars = pd.read_csv(self.gaiastar_filepath)
        self.all_stars = self.all_stars[self.all_stars['phot_g_mean_mag'] <= 16]
        if self.waves_region == 'S':
            self.all_stars = self.all_stars[self.all_stars['dec'] < -10]
        else:
            self.all_stars = self.all_stars[self.all_stars['dec'] > -10]
        self.all_stars['mask_radius'] = mask_radius_waves(self.all_stars['phot_g_mean_mag'])


    def get_stacks(self):
        """
        Compute stacked overdensity around stars using direct offset stacking.
        Uses spherical KDTree for efficient neighbor queries and numba for fast offset computation.
        Now processes each filter separately.
        """
        # Global parameters
        query_extent_factor = 5  # Query radius in units of mask radius
        
        # Process each filter
        for filter_name in self.filters:
            print(f"\n=== Processing filter: {filter_name} ===")
            
            # Apply filter to catalog
            filter_condition = self.filter_conditions[filter_name](self.cat)
            filtered_cat = self.cat[filter_condition].reset_index(drop=True)
            
            print(f"Filtered catalog size: {len(filtered_cat)} (from {len(self.cat)})")
            
            if len(filtered_cat) == 0:
                print(f"No sources after applying filter {filter_name}")
                continue
            
            # Prepare source coordinates for this filter
            sources_coords = SkyCoord(ra=np.array(filtered_cat['RAmax']) * u.deg,
                                    dec=np.array(filtered_cat['Decmax']) * u.deg)
            
            # Convert to radians for spherical calculations
            source_ra_rad = sources_coords.ra.rad
            source_dec_rad = sources_coords.dec.rad
            
            # Create spherical coordinate array for BallTree (lat, lon in radians)
            source_coords_sphere = np.column_stack([source_dec_rad, source_ra_rad])
            
            # Build spherical KDTree using haversine metric (great circle distance)
            print("Building spherical KDTree...")
            tree = BallTree(source_coords_sphere, metric='haversine')
            
            # Initialize stacks for this filter
            if filter_name not in self.stacks:
                self.stacks[filter_name] = {}
                self.bin_edges[filter_name] = {}
            
            for gbin in range(len(self.bin_names)):
                # Select stars in current magnitude bin
                sel = ((self.all_stars['phot_g_mean_mag'] > self.gaia_bins[gbin][0]) &
                    (self.all_stars['phot_g_mean_mag'] <= self.gaia_bins[gbin][1]))
                stars = self.all_stars[sel]
                
                if len(stars) == 0:
                    print(f"No stars in bin {self.bin_names[gbin]}")
                    continue
                    
                # Get star coordinates and radii
                stars_coords = SkyCoord(ra=np.array(stars['ra']) * u.deg,
                                        dec=np.array(stars['dec']) * u.deg)
                radii_arcmin = np.array(stars['mask_radius'])
                
                print(f"Processing bin: {self.bin_names[gbin]}, number of stars: {len(stars)}")
                
                # Set up binning
                nbins = 100
                
                # Create bin edges
                xedges = np.linspace(-self.extent, self.extent, nbins + 1)
                yedges = np.linspace(-self.extent, self.extent, nbins + 1)
                self.bin_edges[filter_name][self.bin_names[gbin]] = (xedges, yedges)
                
                # Initialize stack
                self.stacks[filter_name][self.bin_names[gbin]] = np.zeros((nbins, nbins), dtype=np.float64)
                
                # Process each star
                for i in tqdm(range(len(stars)), desc=f"Stacking {filter_name} - {self.bin_names[gbin]}"):
                    star_ra_rad = stars_coords.ra.rad[i]
                    star_dec_rad = stars_coords.dec.rad[i]
                    mask_radius_arcmin = radii_arcmin[i]
                    
                    # Query radius in radians (great circle distance)
                    query_radius_rad = np.radians(mask_radius_arcmin * query_extent_factor / 60.0)
                    
                    # Query KDTree for nearby sources
                    star_coord_sphere = np.array([[star_dec_rad, star_ra_rad]])
                    indices = tree.query_radius(star_coord_sphere, r=query_radius_rad)[0]
                    
                    if len(indices) == 0:
                        continue
                    
                    # Get coordinates of nearby sources
                    nearby_source_ra = source_ra_rad[indices]
                    nearby_source_dec = source_dec_rad[indices]
                    
                    # Compute angular offsets using numba-accelerated function
                    valid_mask, x_offsets, y_offsets = compute_angular_offsets_numba(
                        star_ra_rad, star_dec_rad, 
                        nearby_source_ra, nearby_source_dec,
                        mask_radius_arcmin, self.extent
                    )
                    
                    # Filter to valid offsets
                    valid_indices = np.where(valid_mask)[0]
                    if len(valid_indices) == 0:
                        continue
                        
                    x_valid = x_offsets[valid_indices]
                    y_valid = y_offsets[valid_indices]
                    
                    # Bin the offsets
                    x_bins = np.digitize(x_valid, xedges) - 1
                    y_bins = np.digitize(y_valid, yedges) - 1
                    
                    # Add to stack (only valid bins)
                    valid_bins = ((x_bins >= 0) & (x_bins < nbins) & 
                                (y_bins >= 0) & (y_bins < nbins))
                    
                    if np.any(valid_bins):
                        x_bins_valid = x_bins[valid_bins]
                        y_bins_valid = y_bins[valid_bins]
                        
                        # Accumulate counts
                        for j in range(len(x_bins_valid)):
                            self.stacks[filter_name][self.bin_names[gbin]][y_bins_valid[j], x_bins_valid[j]] += 1.0
                
                print(f"Completed bin {self.bin_names[gbin]}: total counts = {self.stacks[filter_name][self.bin_names[gbin]].sum()}")
        with open('stacks.pkl', 'wb') as f:
            pickle.dump(self.stacks, f)
        with open('edges.pkl', 'wb') as f:
            pickle.dump(self.bin_edges, f)
        print("Dictionaries saved")


    def load_stacks(self):
        with open('stacks.pkl', 'rb') as f:
            self.stacks = pickle.load(f)
        with open('edges.pkl', 'rb') as f:
            self.bin_edges = pickle.load(f)
        

    def plot_stacks_3x3(self):
        """
        Create a 3x3 plot with filters on rows and magnitude bins on columns
        """
        n_filters = len(self.filters)
        n_bins = len(self.bin_names)
        
        # Create 3x3 subplot
        fig, axes = plt.subplots(n_filters, n_bins, figsize=(5 * n_bins, 5 * n_filters))
        
        # Ensure axes is always 2D
        if n_filters == 1:
            axes = axes.reshape(1, -1)
        if n_bins == 1:
            axes = axes.reshape(-1, 1)
        
        for i, filter_name in enumerate(self.filters):
            for j, bin_name in enumerate(self.bin_names):
                ax = axes[i, j]
                
                # Check if we have data for this combination
                if (filter_name not in self.stacks or 
                    bin_name not in self.stacks[filter_name]):
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=14)
                    ax.set_xlim(-self.extent, self.extent)
                    ax.set_ylim(-self.extent, self.extent)
                    continue
                
                stacked = self.stacks[filter_name][bin_name]
                xedges, yedges = self.bin_edges[filter_name][bin_name]
                
                # Calculate log density
                mean_density = np.mean(stacked[stacked > 0])
                if mean_density > 0:
                    log_density = np.log10(stacked / mean_density)
                    log_density[np.isinf(log_density)] = np.min(log_density[~np.isinf(log_density)])
                else:
                    log_density = np.zeros_like(stacked)
                
                vmax = np.max(np.abs(log_density)) if np.max(np.abs(log_density)) > 0 else 1.0
                
                # Plot
                pcm = ax.pcolormesh(
                    xedges, yedges, log_density,
                    cmap='seismic',
                    shading='flat',
                    vmin=-vmax, vmax=vmax
                )
                


                #Labels and formatting
                if i == n_filters - 1:  # Bottom row
                    ax.set_xlabel(r'$\Delta \mathrm{RA} / R_{\mathrm{mask}}$')
                if j == 0:  # Left column
                    ax.set_ylabel(r'$\Delta \mathrm{Dec} / R_{\mathrm{mask}}$')
                # Title - only show magnitude bin on top row
                if i == 0:  # Top row
                    ax.set_title(f'{bin_name}')

                # Add filter name as row label on the left side
                if j == 0:  # Left column
                    # Position the text to the left of the y-axis
                    ax.text(-0.25, 0.5, filter_name, transform=ax.transAxes, 
                        rotation=90, ha='center', va='center', fontsize=12, fontweight='bold')
                
                ax.set_xlim(-self.extent, self.extent)
                ax.set_ylim(-self.extent, self.extent)
                ax.set_aspect('equal')
                
                # Add circle at mask radius
                circle = patches.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
                ax.add_patch(circle)
                
                # Add colorbar
                cbar = plt.colorbar(pcm, ax=ax)
                cbar.set_label(r'$\log10(\rho/\bar{\rho})$', fontsize=10)
                ax.grid(alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(f'waves-{self.waves_region.lower()}-stacks-3x3-filters.png', dpi=150, bbox_inches='tight')
        plt.show()


    def plot_stacks(self):
        """
        Original plotting function - now plots the first filter only for backward compatibility
        """
        if not self.stacks:
            print("No stacks to plot!")
            return
        
        # Use first filter for backward compatibility
        first_filter = list(self.stacks.keys())[0]
        stacks_to_plot = self.stacks[first_filter]
        bin_edges_to_plot = self.bin_edges[first_filter]
        
        n_bins = len(stacks_to_plot)

        if n_bins <= 2:
            fig, axes = plt.subplots(1, n_bins, figsize=(6 * n_bins, 6))
        elif n_bins <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        else:
            n_rows = (n_bins + 2) // 3
            fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))

        if n_bins == 1:
            axes = [axes]
        elif n_bins > 1 and hasattr(axes, 'flatten'):
            axes = axes.flatten()

        for i, (bin_name, stacked) in enumerate(stacks_to_plot.items()):
            ax = axes[i]

            xedges, yedges = bin_edges_to_plot[bin_name]  

            mean_density = np.mean(stacked[stacked > 0])
            if mean_density > 0:
                log_density = np.log(stacked / mean_density)
                log_density[np.isinf(log_density)] = np.min(log_density[~np.isinf(log_density)])
            else:
                log_density = np.zeros_like(stacked)

            vmax = np.max(np.abs(log_density)) if np.max(np.abs(log_density)) > 0 else 1.0

            pcm = ax.pcolormesh(
                xedges, yedges, log_density,
                cmap='seismic',
                shading='flat',
                vmin=-vmax, vmax=vmax
            )

            ax.set_xlabel(r'$\Delta \mathrm{RA} / R_{\mathrm{mask}}$')
            ax.set_ylabel(r'$\Delta \mathrm{Dec} / R_{\mathrm{mask}}$')
            ax.set_title(f'WAVES-{self.waves_region} Density Around Stars {bin_name} ({first_filter})')
            ax.set_xlim(-self.extent, self.extent)
            ax.set_ylim(-self.extent, self.extent)
            ax.set_aspect('equal')

            circle = patches.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
            ax.add_patch(circle)

            cbar = plt.colorbar(pcm, ax=ax)
            cbar.set_label(r'$\log(\rho/\bar{\rho})$')
            ax.grid(alpha=0.3, linestyle=':')

        if n_bins < len(axes):
            for j in range(n_bins, len(axes)):
                axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'waves-{self.waves_region.lower()}-stacks-{first_filter.lower().replace(" ", "-")}.png', dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    analyzer = star_overdensites(
        # waves_filepath='/Users/sp624AA/Downloads/waves_light/WAVES-S_d1m3p1f1_light.parquet',
        # gaiastar_filepath='/Users/sp624AA/Downloads/Masking/gaiastarmaskwaves.csv',
        waves_region='S',
        gaia_g_bins=[[0, 8], [8,14], [14, 16]]
    )

    print("Loading WAVES data...")
    analyzer.load_waves()

    print("Loading Gaia stars...")
    analyzer.load_gaia_stars()

    print("Calculating stacks...")
    analyzer.get_stacks()

    print("Plotting 3x3 results...")
    analyzer.plot_stacks_3x3()

    print("Analysis complete!")