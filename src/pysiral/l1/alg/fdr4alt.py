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
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import xarray as xr

from pysiral import PSRLCFG
from pysiral.l1 import L1Algorithm, Level1bData
from pysiral.sources.fdr4alt.file_discovery import FDR4ALTFileNaming


class FillInRangeCorrectionsFromTSIPConfig(BaseModel):
    """
    Configuration parameters for the FillInRangeCorrectionsFromTSIP algorithm.
    """
    # Add any configuration parameters here if needed
    local_machine_def_tag: str
    subtag: str
    thematic_product_time_bias_days: float = 0.0
    target_variables: Dict[str, str]  # Mapping of target variable names to their corresponding names in the TSIP files

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
                f"cycle {l1.info.cycle}, time range {l1.info.start_time} to {l1.info.stop_time}."
            )
            return

        for tp_file in tp_file_list:
            logger.info(f"Processing TSIP file: {tp_file}")
            self._fill_in_range_corrections(l1, tp_file)

    def _fill_in_range_corrections(self, l1: Level1bData, tp_file: Path) -> None:
        """

        :param l1: Level-1 data object
        :param tp_file: Path to the Thematic Product file

        :return: None
        """
        target_dict = self.cfg.target_variables
        with xr.open_datatree(tp_file, decode_times=False, decode_cf=True) as ds:

            # Reference variable is the time variable in number notation
            l1_time_num = l1.classifier.time_num
            # NOTE: The ERS-2 thematic product has a time bias of -3652 days (10 years)
            #       compared to the FDR4ALT time variable. This has to be specified in the configuration file.
            tp_time_num = ds.main.time.values + self.cfg.thematic_product_time_bias_days
            index_map_fdr, index_map_tp = self._get_index_mappings(l1_time_num, tp_time_num)

            # NOTE: It happens that file coverage indicates overlap, but the actual data doesn't.
            try:
                assert len(index_map_fdr) > 0, "No overlapping time coverage between L1 data and Thematic Product."
            except AssertionError as e:
                logger.warning(f"{e} Skipping file: {tp_file}")
                return

            # Map the target variables from the Thematic Product to the Level-1 data
            for var_target in target_dict.keys():

                tp_variable_name = target_dict[var_target]

                # NOTE: It happened that individual files missing certain range correction variables.
                try:
                    tp_var = ds[tp_variable_name].values
                except KeyError:
                    logger.warning(f"Could not file range correction {tp_variable_name} in source file")
                    continue

                # Check if the target variable is all NaN
                if np.all(np.isnan(tp_var)):
                    logger.warning(f"Range correction {tp_variable_name} in source file is all NaN")
                    continue

                l1_var = getattr(l1.correction, var_target)
                l1_var[index_map_fdr] = tp_var[index_map_tp]
                setattr(l1.correction, var_target, l1_var)

    @staticmethod
    def _get_index_mappings(
            l1_time_num: np.ndarray,
            tp_time_num: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the indices arrays that map the L1 data coverage with the
        files from the FDR4ALT thematic in the sense of

            l1_time[l1_index_map] = tp_time[tp_index_map]

        Implementation from
        https://stackoverflow.com/questions/57873670/index-mapping-between-two-sorted-partially-overlapping-numpy-arrays

        :param l1_time_num: The Level-1 (FDR) time as numeric value
        :param tp_time_num: The Thematic product time as numeric value

        :return: (l1_index_map, tp_index_map)
        """

        # Convert to integer (milliseconds since epoch) for safe comparison
        # (default date units for FDR4ALT is days since epoch)
        days_to_milliseconds = 86_000.0 * 1000.0
        l1_time_int = (l1_time_num * days_to_milliseconds).astype(int)
        tp_time_int = (tp_time_num * days_to_milliseconds).astype(int)

        # Use numpy's `intersect1d` to find the common elements and their indices
        _, l1_index_map, tp_index_map = np.intersect1d(l1_time_int, tp_time_int, return_indices=True)
        # Validate the mapping
        assert len(l1_index_map) == len(tp_index_map), "Index maps must be of the same length."
        assert np.array_equal(l1_time_int[l1_index_map], tp_time_int[tp_index_map]), "Mapped times must be equal."

        return l1_index_map, tp_index_map

    def query_tp_files(self, platform: str, cycle_number: int, tcs: datetime, tce: datetime) -> Optional[List[Path]]:
        """
        Query the TSIP files for the given time range.

        :param platform: Platform name
        :param cycle_number: Cycle number of the query
        :param tcs: Start time of the query
        :param tce: End time of the query
        """
        file_catalog = self.get_cycle_catalog(platform, cycle_number)
        if file_catalog.empty:
            return None

        # Some sanity checks on the file catalog
        # assert file_catalog.esa_file_type.nunique() == 1, "Multiple ESA file type IDs found in thematic file catalog."
        file_types = file_catalog.esa_file_type.unique()
        assert all('F4A_ALT_TDP_SI_' in ft for ft in file_types), "File type != 'F4A_ALT_TDP_SI_*'"

        # Check for overlapping time coverage with the query period
        min_starts = file_catalog.time_coverage_start.apply(lambda x: max(x, tcs))
        min_ends = file_catalog.time_coverage_end.apply(lambda x: min(x, tce))

        # Compute overlap in seconds (positive & non-zero means overlap exists)
        # NOTE: This is done because the time coverage indicated by the filename is only with second resolution
        #       and this can lead to false negatives due to integer second rounding errors
        overlap_seconds = (min_ends - min_starts).values.astype('timedelta64[s]')
        query_subset = file_catalog[overlap_seconds > 1]
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
        cycle_directories = lookup_directory.rglob(f"**/{cycle_number:d}")
        file_list = []
        for cycle_directory in cycle_directories:
            file_list.extend(cycle_directory.rglob("*.nc"))
        return pd.DataFrame([FDR4ALTFileNaming(f) for f in sorted(file_list)])
