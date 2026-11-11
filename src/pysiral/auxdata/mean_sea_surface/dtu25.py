# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import xarray as xr
import scipy.ndimage as ndimage

from loguru import logger
from time import time

from pysiral.auxdata import AuxDataLoader, AuxDataExitFlag



class DTU25Grid(AuxDataLoader):
    """
    Parsing Routine for DTU 1 km global mean sea surface height files
    """

    def __init__(self, cfg):

        super(DTU25Grid, self).__init__()
        self.cfg = cfg

        # Read as standard netcdf
        t0 = time()
        dtu_grid = xr.load_dataset(self.cfg.filename)
        t1 = time()
        logger.debug(f"Loaded DTU25 dataset in : {t1 - t0:.3f} seconds")

        # Cut to ROI regions (latitude only)
        # -> no need for world mss
        lat_range = self.cfg.options.latitude_range

        # Get the indices for the latitude subset
        latitude_indices = np.where(
            np.logical_and(
                dtu_grid.latitude.values >= lat_range[0],
                dtu_grid.latitude.values <= lat_range[1]
            )
        )[0]

        # Crop data to subset
        self.elevation = dtu_grid.mss.values[latitude_indices, :]
        self.longitude = dtu_grid.longitude.values - 180.  # DTU MSS25 longitude is 0-360, convert to -180 to 180
        self.latitude = dtu_grid.latitude.values[latitude_indices]

    def apply(self, l1, l2):

        # Use fast image interpolation (since DTU is on regular grid)
        # Longitudes must be 0 -> 360

        longitude = np.array(l2.track.longitude)
        latitude = np.array(l2.track.latitude)

        # Calculate image coordinates of mss grid "image"
        mss_lon_min = self.longitude[0]
        mss_lon_step = self.longitude[1] - self.longitude[0]
        mss_lat_min = self.latitude[0]
        mss_lat_step = self.latitude[1] - self.latitude[0]
        ix = (longitude - mss_lon_min)/mss_lon_step
        iy = (latitude - mss_lat_min)/mss_lat_step

        # Extract and return the elevation along the track
        mss_track_elevation = ndimage.map_coordinates(self.elevation, [iy, ix])

        # Register auxdata variable
        l2.set_auxiliary_variable("mss", "mean_sea_surface", mss_track_elevation, None)
        return AuxDataExitFlag.NOMINAL