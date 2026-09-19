# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt
import xarray as xr

from pysiral import psrlcfg
from pysiral.core.flags import SURFACE_TYPE_DICT
from pysiral.core.grid import GridDefinition, GridTrajectoryExtract
from pysiral.l1.l1data import Level1bData
from pysiral.l1.alg import L1PProcItem


class L1PHighResolutionLandMask(L1PProcItem):
    """
    Level-1 processor item providing access to a high resolution
    land mask and distance to land fields
    """

    def __init__(self, **cfg):
        """
        Initialize the class. This step includes parsing the static mask
        and keeping it in memory
        :param cfg:
        """
        super(L1PHighResolutionLandMask, self).__init__(**cfg)

        # Get the local file path
        type_ = self.cfg.get("local_machine_def_auxclass")
        tag = self.cfg.get("local_machine_def_tag")
        filename = self.cfg.get("filename")
        lookup_directory = psrlcfg.local_machine.auxdata_repository[type_][tag]

        # Set the file for each hemisphere type
        mask_filepath = Path(lookup_directory) / str(filename)
        with xr.open_dataset(mask_filepath) as nc:
            self.grid_def = self.cfg.get("grid_def")
            self.land_ocean_flag_grid = nc.land_ocean_flag.values
            self.distance_to_coast_grid = nc.distance_to_coast.values

    def apply(self, l1: Level1bData) -> None:
        """
        Extract land/ocean flag and distance to coast along the l1p trajectory if a mask exists
        for the corresponding hemisphere of the l1p data object.

        The parameters are stored in the classifier data group among the original surface type
        value, which is then update for the mask coverage

        :param l1:
        :return: None
        """

        # Get the land/ocean flag from the grid
        land_ocean_flag, distance_to_coast = self.get_trajectory(l1.time_orbit.longitude, l1.time_orbit.latitude)

        # TODO: Update when global land/ocean flag becomes available
        # NOTE: Currently only a grid for the northern hemisphere is available, thus
        #       there will be regions not covered by the gridded land/ocean flag.
        #       That's why the built/in land/ocean flag will only be updated for
        #       region with coverage and be left untouched everywhere else.
        #       The arrays are added to the l1 classifier container regardless
        #       for consistency.

        # --- Update the L1 data container ---
        # 1. Save both extracted variables to classifier data groups
        l1.classifier.add(land_ocean_flag, "hr_land_ocean_flag")
        l1.classifier.add(distance_to_coast, "distance_to_coast")

        # 2. Save original ESA surface type variable to classifier data group
        #    and update the surface type flag in the surface type data group
        l1.classifier.add(l1.surface_type.flag, "orig_land_ocean_flag")

        # 3. Update the surface type instance
        valid_mask_indices = land_ocean_flag != self.cfg.get("dummy_val")["land_ocean_flag"]
        flag_update = np.full(l1.n_records, SURFACE_TYPE_DICT["invalid"])
        flag_update[land_ocean_flag == 1] = SURFACE_TYPE_DICT["land"]
        flag_update[land_ocean_flag == 0] = SURFACE_TYPE_DICT["ocean"]
        updated_surface_type_flag = l1.surface_type.flag.copy()
        updated_surface_type_flag[valid_mask_indices] = flag_update[valid_mask_indices]
        l1.surface_type.set_flag(updated_surface_type_flag)

    def get_trajectory(self, longitude: npt.NDArray, latitude: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Get extract of the land/ocean flag and the distance to coast value for an
        array of (longitude, latitude) positions.

        If longitude, latitude is outside the grid, a pre-defined dummy value will
        be returned.

        :param longitude: Longitude values in degrees
        :param latitude: latitude values in degrees

        :raises None:

        :return: land ocean flag & distance to coast values for longitude, latitude positions
        """
        grid2track = GridTrajectoryExtract(longitude, latitude, self.grid_def)
        land_ocean_flag = grid2track.get_from_grid_variable(
            self.land_ocean_flag_grid,
            outside_value=self.dummy_val["land_ocean_flag"]
        )

        distance_to_coast = grid2track.get_from_grid_variable(
            self.distance_to_coast_grid,
            outside_value=self.dummy_val["distance_to_coast"]
        )
        return land_ocean_flag, distance_to_coast