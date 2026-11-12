# -*- coding: utf-8 -*-

from loguru import logger
from pathlib import Path

import re
import numpy as np

from pysiral.auxdata import GridTrackInterpol, AuxDataLoader, AuxDataExitFlag
from pysiral.auxdata.snow_depth._data import SnowParameterContainer
from pysiral.core.iotools import ReadNC
from pysiral.l2.alg.filter import idl_smooth


class ICDCSouthernClimatology(AuxDataLoader):
    """ Class for daily climatology fields from UHH ICDC """

    def __init__(self, cfg):
        self.cfg = cfg
        super(ICDCSouthernClimatology, self).__init__()
        self._data = None

    def apply(self, l1, l2):

        exit_status = AuxDataExitFlag.NOMINAL

        # Set the requested data
        self.set_requested_date_from_l2(l2)

        # CAVEAT: The method `update_external_data()` will fail
        # if the requested date is February 29, since no
        # corresponding source file exists. As a fix, the data in
        # in this case is set back to the February 28.
        if self.month == "02" and self.day == "29":
            self.set_requested_date(int(self.year), int(self.month), 28)

        # Update the external data
        self.update_external_data()

        # Check if error with file I/O
        if self._data is None:
            # This will return an empty container
            snow = SnowParameterContainer()
            snow.set_dummy(l2.n_records)
            exit_status = AuxDataExitFlag.ERROR_CONTINUE
        else:
            # Extract along track snow depth and density
            sd, sd_unc = self._get_snow_track(l2)

            # Apply along-track smoothing if required
            smooth_snowdepth = self.cfg.options.get("self.cfg.options", False)
            if smooth_snowdepth:
                filter_width = self.cfg.options.smooth_filter_width_m
                # Convert filter width to index
                filter_width /= l2.footprint_spacing
                # Round to odd number
                filter_width = np.floor(filter_width) // 2 * 2 + 1
                sd = idl_smooth(sd, filter_width)
                sd_unc = idl_smooth(sd_unc, filter_width)

            # Collect Parameters and return
            # (density and density uncertainty fixed from l2 settings)
            snow = SnowParameterContainer()
            snow.depth = sd
            snow.depth_uncertainty = sd_unc
            snow.density = np.full(sd.shape, self.cfg.options.snow_density)
            snow.density_uncertainty = np.full(sd.shape, self.cfg.options.snow_density_uncertainty)

        # Register Variables
        l2.set_auxiliary_parameter("sd", "snow_depth", snow.depth, snow.depth_uncertainty)
        l2.set_auxiliary_parameter("sdens", "snow_density", snow.density, snow.density_uncertainty)

        return exit_status

    def load_requested_auxdata(self):
        """ Loads file from local repository only if needed """

        # Retrieve the file path for the requested date from a property of the auxdata parent class
        path = Path(self.requested_filepath)

        # Validation
        if not path.is_file():
            msg = "%s: File not found: %s " % (self.__class__.__name__, path)
            logger.error(msg)
            return AuxDataExitFlag.ERROR_CONTINUE

        # Store the netCDF data object
        self._data = ReadNC(path)
        return AuxDataExitFlag.NOMINAL

    def _get_snow_track(self, l2):
        """ Extract snow depth from grid """

        # Extract from grid
        hemisphere_code_dict = {"nh": "north", "sh": "south"}
        hemisphere_code = hemisphere_code_dict.get(l2.hemisphere, l2.hemisphere)
        griddef = self.cfg.options[hemisphere_code]
        grid_lons, grid_lats = self._data.lon, self._data.lat
        grid2track = GridTrackInterpol(l2.track.longitude, l2.track.latitude, grid_lons, grid_lats, griddef)

        # Extract snow depth along track data from grid
        sd_parameter_name = self.cfg.options.snow_depth_nc_variable
        sdgrid = getattr(self._data, sd_parameter_name)[0, :, :]
        snow_depth = grid2track.get_from_grid_variable(sdgrid, flipud=True)
        snow_depth[snow_depth < 0.0] = np.nan

        # Extract snow depth uncertainty
        unc_parameter_name = self.cfg.options.snow_depth_uncertainty_nc_variable
        uncgrid = getattr(self._data, unc_parameter_name)[0, :, :]
        snow_depth_uncertainty = grid2track.get_from_grid_variable(uncgrid, flipud=True)
        snow_depth_uncertainty[snow_depth_uncertainty < 0.0] = np.nan

        return snow_depth, snow_depth_uncertainty

    @property
    def requested_filepath(self) -> "Path":
        """
        Returns the local file path for the requested date
        :return:
        """

        # Main directory
        path = Path(self.cfg.local_repository)

        # Add the sub-folders
        for subfolder_tag in self.cfg.subfolders:
            subfolder = getattr(self, subfolder_tag)
            path = path / subfolder

        # Get the period dict (will be constructed from filenaming)
        period_dict = {}
        attrs = re.findall("{.*?}", self.cfg.filenaming)
        for attr_def in attrs:
            attr_name = attr_def[1:-1]
            period_dict[attr_name] = getattr(self, attr_name)
        filename = self.cfg.filenaming.format(**period_dict)
        path = path / filename
        return path