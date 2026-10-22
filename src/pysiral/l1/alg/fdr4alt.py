# -*- coding: utf-8 -*-

"""
This module contains Level-1 Pre-Processor algorithms specifically for the FDR4ALT dataset.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


from functools import lru_cache
from pydantic import BaseModel
from datetime import datetime
from loguru import logger
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import cftime

from pysiral import PSRLCFG
from pysiral.l1 import L1Algorithm, Level1bData
from pysiral.mission.envisat.file_discovery import FDR4ALTFileNamingFDR


class FillInRangeCorrectionsFromTSIPConfig(BaseModel):
    """
    Configuration parameters for the FillInRangeCorrectionsFromTSIP algorithm.
    """
    # Add any configuration parameters here if needed
    local_machine_def_tag: str
    subtag: str


    def get_lookup_dir(self, platform: str) -> str:
        """
        Compute the lookup directory based on the local machine definition tag and subtag.

        :return: Path to the lookup directory
        """
        primary_input_def = PSRLCFG.local_machine.get("l1b_repository")
        return primary_input_def[platform][self.local_machine_def_tag][self.subtag]

class FillInRangeCorrectionsFromTSIP(L1Algorithm):

    def __init__(self, **kwargs) -> None:
        """
        Level-1 Pre-Processor algorithm to fill in missing range corrections from TSIP data.

        :param kwargs: Configuration parameters for the algorithm
        """
        self.cfg = FillInRangeCorrectionsFromTSIPConfig(**kwargs)


    def apply(self, l1):
        """
        Apply the algorithm to the Level-1 data.

        :param l1: Level-1 data object
        """

        # Can be several two thematic sea ice files
        tp_file_list = self.query_tp_files(
            l1.info.mission,
            l1.info.cycle,
            l1.info.start_time,
            l1.info.stop_time
        )
        if tp_file_list is None:
            logger.warning(
                f"No Thematics sea ice files files found for platform {l1.info.mission}, "
                f"cycle {l1.info.cycle_number}, time range {l1.info.start_time} to {l1.info.stop_time}."
            )
            return

        for tp_file in tp_file_list:
            logger.info(f"Processing TSIP file: {tp_file}")
            self._fill_in_range_corrections(l1, tp_file)

    def _fill_in_range_corrections(self, l1: Level1bData, tp_file: Path) -> None:

        with xr.open_datatree(tp_file, decode_times=False) as ds:

            index_map_fdr, index_map_tp = self._get_index_map(l1.time_orbit.timestamp, ds.main.time)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.scatter(ds.main.time.values, ds.main.time.values, s=64, label="TSIP")
            time_l1 = cftime.date2num(l1.time_orbit.timestamp, units=ds.main.time.attrs["units"])
            plt.scatter(time_l1, time_l1, s=12, label="L1")
            plt.show()
            breakpoint()

    def _get_index_map(
            self,
            l1_time_dt: np.ndarray,
            tp_time_num: xr.Variable
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the indices arrays that map the L1 data coverage with the
        files from the FDR4ALT thematic in the sense of

            l1_time[l1_index_map] = tp_time[tp_index_map]

        Implementation from https://stackoverflow.com/questions/57873670/index-mapping-between-two-sorted-partially-overlapping-numpy-arrays

        :param l1_time_dt: The Level-1 timestamp as datetime object
        :param tp_time_num: The time num from the thematic products

        :return: (l1_index_map, tp_index_map)
        """

        # First, convert the L1 time stamp to the same date numbers
        l1_time_num = cftime.date2num(l1_time_dt, units=tp_time_num.attrs["units"])

        # Convert to integer for safe comparison

        l1_time_int = (l1_time_num * 10_000).astype(int)
        tp_time_int = (tp_time_num.values * 10_000).astype(int)

        over_lap_mask = np.isin(tp_time_int, l1_time_int)
        tp_index_map = np.where(over_lap_mask)

        mapping = np.searchsorted(l1_time_int, tp_time_int)
        l1_index_map = np.where(mapping >= len(l1_time_int))
        breakpoint()

    def query_tp_files(self, platform: str, cycle_number: int, tcs: datetime, tce: datetime) -> Optional[List[Path]]:
        """
        Query the TSIP files for the given time range.

        :param platform: Platform name
        :param cycle_number: Cycle number of the query
        :param tcs: Start time of the query
        :param tce: End time of the query
        """
        file_catalog = self.get_cycle_catalog(platform, cycle_number)

        # Check for overlapping time coverage with the query period
        min_starts = file_catalog.time_coverage_start.apply(lambda x: max(x, tcs))
        min_ends = file_catalog.time_coverage_end.apply(lambda x: min(x, tce))
        query_subset = file_catalog[min_starts < min_ends]
        return query_subset.filepath.to_list() if not query_subset.empty else None

    @lru_cache(maxsize=256)
    def get_cycle_catalog(self, platform: str, cycle_number: int) -> pd.DataFrame:
        """
        Get the catalog of TSIP files for the given cycle number and platform.
        NOTE: It is anticipated that the cycle directories may be distributed over two years, also
              that the query will be repeated several times for the same cycle number, hence the use of caching.

        :param platform: Platform name
        :param cycle_number: Cycle number of the query
        :return: DataFrame containing TSIP file information
        """
        primary_input_def = PSRLCFG.local_machine.get("l1b_repository")
        lookup_directory = Path(primary_input_def[platform][self.cfg.local_machine_def_tag]["source"][self.cfg.subtag])
        assert lookup_directory.is_dir(), f"Lookup directory {lookup_directory} does not exist or is not a directory."

        # Cycle directories may be distributed over two years
        cycle_directories = lookup_directory.rglob(f"**{cycle_number:03d}")
        file_list = []
        for cycle_directory in cycle_directories:
            file_list.extend(cycle_directory.rglob("*.nc"))
        return pd.DataFrame([FDR4ALTFileNamingFDR(f) for f in sorted(file_list)])


