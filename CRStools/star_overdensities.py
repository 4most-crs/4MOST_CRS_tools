import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def mask_radius_waves(g):
    g = np.asarray(g)
    r = np.zeros_like(g, dtype=float)
    mask1 = (g > 3.5) & (g < 16)
    mask2 = g <= 3.5
    r[mask1] = (10 ** (1.3 - 0.13 * g[mask1]))
    r[mask2] = 7
    return r


class star_overdensites:
    def __init__(self,
                 waves_filepath='/Users/sp624AA/Downloads/Ultralight/waves_s_ultralite.parquet',
                 gaiastar_filepath='/Users/sp624AA/Downloads/Masking/gaiastarmaskwaves.csv',
                 waves_region='S',
                 gaia_g_bins=[[0, 8], [8, 13], [13, 15], [15, 16]]):

        self.waves_filepath = waves_filepath
        self.gaiastar_filepath = gaiastar_filepath
        self.waves_region = waves_region

        if self.waves_region not in ['N', 'S']:
            raise ValueError('waves_region must be either N or S')

        self.gaia_bins = gaia_g_bins
        self.bin_names = [f"{b[0]}<G<{b[1]}" for b in self.gaia_bins]
        print(f'bin names: {self.bin_names}')

        self.stacks = {}
        self.bin_edges = {}  # Holds xedges, yedges for each bin
        self.nside = 1024 * 32
        self.extent = 3  # R units
        self._setup_healpix_region()

    def _setup_healpix_region(self):
        if self.waves_region == 'S':
            ra_min, ra_max = np.radians([330.0, 411.6])
            dec_min, dec_max = np.radians([-35.6, -27.0])
        else:
            ra_min, ra_max = np.radians([157.25, 225.0])
            dec_min, dec_max = np.radians([-3.95, 3.95])

        self.polygon = np.array([
            [ra_min, dec_min],
            [ra_max, dec_min],
            [ra_max, dec_max],
            [ra_min, dec_max]
        ])

        vecs = hp.ang2vec(np.pi / 2 - self.polygon[:, 1], self.polygon[:, 0])
        region_pix = hp.query_polygon(self.nside, vecs)
        self.region_pix = set(region_pix)

    def load_waves(self):
        self.cat = pq.read_table(self.waves_filepath).to_pandas()
        self.cat = self.cat[(self.cat['duplicate'] == 0)]

    def load_gaia_stars(self):
        self.all_stars = pd.read_csv(self.gaiastar_filepath)
        self.all_stars = self.all_stars[self.all_stars['phot_g_mean_mag'] <= 16]
        if self.waves_region == 'S':
            self.all_stars = self.all_stars[self.all_stars['dec'] < -10]
        else:
            self.all_stars = self.all_stars[self.all_stars['dec'] > -10]
        self.all_stars['mask_radius'] = mask_radius_waves(self.all_stars['phot_g_mean_mag'])

    def get_stacks(self):
        # Prepare coordinate transforms
        sources_coords = SkyCoord(ra=np.array(self.cat['RAmax']) * u.deg,
                                  dec=np.array(self.cat['Decmax']) * u.deg)

        ra_rad = sources_coords.ra.rad
        dec_rad = sources_coords.dec.rad
        ipix_sources = hp.ang2pix(self.nside, 0.5 * np.pi - dec_rad, ra_rad)

        mask = np.isin(ipix_sources, list(self.region_pix))
        ipix_sources = ipix_sources[mask]
        density_map = Counter(ipix_sources)

        # Use global minimum radius to set binning resolution
        min_radius = self.all_stars['mask_radius'].min()
        res_arcmin = hp.nside2resol(self.nside, arcmin=True)
        nbins = int(np.floor(self.extent * min_radius / res_arcmin))
        print(f"NSIDE = {self.nside}, resolution ≈ {res_arcmin:.2f} arcmin")
        print("Global min radius:", min_radius)
        print("Using nbins =", nbins)

        self.nbins = nbins
        self.xedges = np.linspace(-self.extent, self.extent, nbins + 1)
        self.yedges = np.linspace(-self.extent, self.extent, nbins + 1)

        rad_to_arcmin = 180 / np.pi * 60
        query_extent = 5  # units of R

        for gbin in range(len(self.bin_names)):
            sel = ((self.all_stars['phot_g_mean_mag'] > self.gaia_bins[gbin][0]) &
                (self.all_stars['phot_g_mean_mag'] <= self.gaia_bins[gbin][1]))
            stars = self.all_stars[sel]
            stars_coords = SkyCoord(ra=np.array(stars['ra']) * u.deg,
                                    dec=np.array(stars['dec']) * u.deg)
            radii = np.array(stars['mask_radius'])

            print(f"Processing bin: {self.bin_names[gbin]}, number of stars: {len(stars)}")

            min_radius = np.min(radii)
            nbins = int(np.floor(self.extent * min_radius / res_arcmin))
            print(f"Bin {self.bin_names[gbin]}: min radius = {min_radius:.2f} arcmin, nbins = {nbins}")

            xedges = np.linspace(-self.extent, self.extent, nbins + 1)
            yedges = np.linspace(-self.extent, self.extent, nbins + 1)
            self.bin_edges[self.bin_names[gbin]] = (xedges, yedges)

            self.stacks[self.bin_names[gbin]] = np.zeros((nbins, nbins), dtype=np.float32)

            for i in tqdm(range(len(stars))):
                ra0 = stars_coords.ra.rad[i]
                dec0 = stars_coords.dec.rad[i]
                R_arcmin = radii[i]
                R_deg = R_arcmin * query_extent / 60.0

                vec = hp.ang2vec(0.5 * np.pi - dec0, ra0)
                ipix_disc = hp.query_disc(self.nside, vec, np.radians(R_deg), inclusive=True)

                for ipix in ipix_disc:
                    if ipix not in density_map:
                        continue

                    theta, phi = hp.pix2ang(self.nside, ipix)
                    ra_pix = phi
                    dec_pix = 0.5 * np.pi - theta

                    dra = (ra_pix - ra0) * np.cos(dec0)
                    ddec = dec_pix - dec0

                    x = (dra * rad_to_arcmin) / R_arcmin
                    y = (ddec * rad_to_arcmin) / R_arcmin

                    if abs(x) < self.extent and abs(y) < self.extent:
                        ix = np.searchsorted(xedges, x) - 1
                        iy = np.searchsorted(yedges, y) - 1
                        if 0 <= ix < nbins and 0 <= iy < nbins:
                            self.stacks[self.bin_names[gbin]][iy, ix] += density_map[ipix]

    def plot_stacks(self):
        n_bins = len(self.stacks)

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

        for i, (bin_name, stacked) in enumerate(self.stacks.items()):
            ax = axes[i]

            xedges, yedges = self.bin_edges[bin_name]  

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
            ax.set_title(f'WAVES-{self.waves_region} Density Around Stars {bin_name}')
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
        plt.savefig(f'waves-{self.waves_region.lower()}-all-bins-stacks.png', dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    analyzer = star_overdensites(
        waves_filepath='/Users/sp624AA/Downloads/Ultralight/waves_s_ultralite.parquet',
        gaiastar_filepath='/Users/sp624AA/Downloads/Masking/gaiastarmaskwaves.csv',
        waves_region='S',
        gaia_g_bins=[[0, 8], [8, 13], [13, 15], [15, 16]]
    )

    print("Loading WAVES data...")
    analyzer.load_waves()

    print("Loading Gaia stars...")
    analyzer.load_gaia_stars()

    print("Calculating stacks...")
    analyzer.get_stacks()

    print("Plotting results...")
    analyzer.plot_stacks()

    print("Analysis complete!")
