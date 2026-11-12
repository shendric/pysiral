# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import xarray as xr
import scipy.ndimage as ndimage

from pysiral.auxdata import AuxDataLoader, AuxDataExitFlag
from pysiral.auxdata.mean_sea_surface.functions import egm2top_delta_h, egm2wgs_delta_h


class DTU1MinGrid(AuxDataLoader):
    """
    Parsing Routine for DTU 1 minute global mean sea surface height files
    """

    def __init__(self, cfg, **kwargs):

        super(DTU1MinGrid, self).__init__()

        self.cfg = cfg

        # Read as standard netcdf
        dtu_grid = xr.load_dataset(cfg.filename)

        # Cut to ROI regions (latitude only)
        # -> no need for world mss
        lat_range = self.cfg.options.latitude_range

        # Get the indices for the latitude subset
        latitude_indices = np.where(np.logical_and(dtu_grid.lat >= lat_range[0], dtu_grid.lat <= lat_range[1]))[0]

        # Crop data to subset
        self.elevation = dtu_grid.mss[latitude_indices, :]
        self.longitude = dtu_grid.lon
        self.latitude = dtu_grid.lat[latitude_indices]

        # Convert elevations to WGS84
        delta_h1 = egm2top_delta_h(self.latitude)
        delta_h = egm2wgs_delta_h(self.latitude)
        for i in np.arange(len(self.latitude)):
            self.elevation[i, :] += (delta_h[i]-delta_h1[i])

    def apply(self, l1, l2):

        # Use fast image interpolation (since DTU is on regular grid)
        # Longitudes must be 0 -> 360

        longitude = np.array(l2.track.longitude)
        latitude = np.array(l2.track.latitude)

        negative_lons = np.where(longitude < 0)[0]
        longitude[negative_lons] = longitude[negative_lons] + 360.

        # Calculate image coordinates of mss grid "image"
        mss_lon_min = self.longitude.values[0]
        mss_lon_step = self.longitude.values[1] - self.longitude.values[0]
        mss_lat_min = self.latitude.values[0]
        mss_lat_step = self.latitude.values[1] - self.latitude.values[0]
        ix = (longitude - mss_lon_min)/mss_lon_step
        iy = (latitude - mss_lat_min)/mss_lat_step

        # Extract and return the elevation along the track
        mss_track_elevation = ndimage.map_coordinates(self.elevation.values, [iy, ix])

        # Register auxdata variable
        l2.set_auxiliary_parameter("mss", "mean_sea_surface", mss_track_elevation, None)
        return AuxDataExitFlag.NOMINAL