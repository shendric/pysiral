# -*- coding: utf-8 -*-

import re
from typing import Dict, List, Literal
from collections import deque
from pathlib import Path

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from pysiral.l1preproc import Level1InputHandler


class CS2ICEFileDiscoveryConfig(BaseModel):
    """
    Configuration for ESA CryoSat-2 ICE File discovery.
    (Default values for baseline-E).

    """
    lookup_modes: List[Literal["sar", "sin", "lrm"]] = Field(
        default=["sar", "sin"],
        description="List of radar modes"
    )
    filename_search: str = Field(
        default="CS_*_SIR_*1B_{year:04d}{month:02d}{day:02d}*_E*.nc",
        description="File search pattern"
    )
    filename_sep: str = Field(
        default="_",
        description="character to split the filename"
    )
    tcs_str_index: int = Field(
        default=5,
        description="Index of file name item indicating the start date"
    )


class CS2ICEFileDiscovery(
    Level1InputHandler,
    supports=[
        "cryosat2_rep_esa_ice_b00E",
        "cryosat2_nrt_esa_ice_b00E",
    ]
):
    """
    Class for file discovery of CryoSat-2 ICE Level-1b data products (SAR, SIN, LRM).

    The class expects as in
    """

    def __init__(
            self,
            lookup_directory: Dict[str, Path],
            options_kwargs: Dict
    ) -> None:
        """
        Initialize the

        :param lookup_directory:
        :param options_kwargs:
        """
        self._lookup_directory = lookup_directory
        self.cfg = CS2ICEFileDiscoveryConfig(**options_kwargs)

    def get_file_for_period(self, period):
        """ Return a list of sorted files """
        # Make sure file list are empty
        self._reset_file_list()
        for mode in self.cfg.lookup_modes:
            self._append_files(mode, period)
        return self.sorted_list

    def _reset_file_list(self):
        self._list = deque([])
        self._sorted_list = []

    def _append_files(self, mode, period):
        lookup_year, lookup_month = period.tcs.year, period.tcs.month
        lookup_dir = self._get_lookup_dir(lookup_year, lookup_month, mode)
        logger.info(f"Search directory: {lookup_dir}")
        n_files = 0
        for daily_period in period.get_segments("day"):
            # Search for specific day
            year, month, day = daily_period.tcs.year, daily_period.tcs.month, daily_period.tcs.day
            file_list = self._get_files_per_day(lookup_dir, year, month, day)
            tcs_list = self._get_tcs_from_filenames(file_list)
            n_files += len(file_list)
            for file, tcs in zip(file_list, tcs_list):
                self._list.append((file, tcs))
        logger.info(" Found %g %s files" % (n_files, mode))

    def _get_files_per_day(self, lookup_dir, year, month, day):
        """ Return a list of files for a given lookup directory """
        # Search for specific day
        filename_search = self.cfg.filename_search.format(year=year, month=month, day=day)
        return sorted(Path(lookup_dir).glob(filename_search))

    def _get_lookup_dir(self, year, month, mode):
        yyyy, mm = "%04g" % year, "%02g" % month
        return Path(self.cfg.lookup_dir[mode]) / yyyy / mm

    def _get_tcs_from_filenames(self, files):
        """
        Extract the part of the filename that indicates the time coverage start (tcs)
        :param files: a list of files
        :return: tcs: a list with time coverage start strings of same length as files
        """
        tcs = []
        for filename in files:
            filename_segments = re.split(r"_+|\.", str(Path(filename).name))
            tcs.append(filename_segments[self.cfg.tcs_str_index])
        return tcs

    @property
    def lookup_directory(self):
        if isinstance(self._lookup_directory, Path):
            return Path(self._lookup_directory)
        elif isinstance(self._lookup_directory, dict):
            return dict(**self._lookup_directory)
        raise TypeError(f"{self._lookup_directory=} neither Path nor dict (This should not happen)")

    @property
    def sorted_list(self):
        dtypes = [('path', object), ('start_time', object)]
        self._sorted_list = np.array(self._list, dtype=dtypes)
        self._sorted_list.sort(order='start_time')
        return [item[0] for item in self._sorted_list]
