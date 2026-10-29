# -*- coding: utf-8 -*-

"""
This module implements the Envisat FDR4ALT file discovery class.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = ["FDR4ALTFileDiscovery"]

from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
from pydantic import BaseModel
from typing import Dict, Optional, List
from parse import parse
from pathlib import Path

import pandas as pd

from pysiral.sources._io_abc import FileDiscovery


@dataclass
class FDR4ALTFileNaming:
    """
    Deciphering the Envisat FDR4ALT filenaming convention
    (source: Envisat FDR4ALT File Naming Convention)
    """
    filepath: Path
    is_valid_file: bool = field(init=False)
    file_id: str = field(init=False)
    mission_id: str = field(init=False)
    esa_file_type: str = field(init=False)
    time_coverage_start: datetime = field(init=False)
    time_coverage_end: datetime = field(init=False)
    cycle: int = field(init=False)
    pass_number: int = field(init=False)
    extension: str = field(init=False)

    def __post_init__(self):

        # Define the filenaming convention
        # EN1_F4A_ALT_FDR____101_0009_20110327T041659_20110327T050706_V01.nc
        filenaming_convention = (
            "{mission_id:3}_{esa_file_type:10}_{cycle:3d}_{pass_number:4d}_"
            "{time_coverage_start:%Y%m%dT%H%M%S}_{time_coverage_end:%Y%m%dT%H%M%S}_V{version:2d}.{extension:2}"
        )

        # The file is the name of the last part of the path
        self.file_id = self.filepath.parent.parts[-1]

        # Parse the file id
        elements = parse(filenaming_convention, self.filepath.name)
        if elements is None:
            logger.error(f"{self.file_id} is not a valid FDR4ALT filename [{filenaming_convention}]")
            self.is_valid_file = False
            return
        self.is_valid_file = True

        # Assign the parsed string properties
        for key in elements.named.keys():
            setattr(self, key, elements.named[key])


class FDR4ALTFileDiscoveryConfig(BaseModel):
    local_machine_def_tag: str
    subtag: str
    lookup_dir: Dict[str, Path]

    def get_lookup_dir(self, year: int) -> Optional[Path]:
        """
        Get the lookup directory for a given year.
        """

        directory = Path(self.lookup_dir[self.subtag]) / f"{year}"
        return directory if directory.is_dir() else None


class FDR4ALTFileDiscovery(FileDiscovery):

    def __init__(self, **kwargs):
        self.cfg = FDR4ALTFileDiscoveryConfig(**kwargs)

    def query_period(self, period) -> Optional[pd.DataFrame]:
        """
        Abstract method to load source data into Level 1 data object from a given file path.
        """
        file_catalog = self._get_year_catalog(period)
        if file_catalog.empty:
            return None
        query_subset = file_catalog[
            (file_catalog.time_coverage_start >= period.tcs.dt) &
            (file_catalog.time_coverage_end <= period.tce.dt)
            ]
        query_subset.reset_index(inplace=True)
        return query_subset if not query_subset.empty else None

    def _get_year_catalog(self, period) -> pd.DataFrame:
        """
        Get a file catalog for the years listed in the period query.

        :param period: The period to query for.

        :return: A pandas DataFrame containing all netCDF within the query years.
        """
        yearly_periods = list(period.get_segments("P1Y"))
        years = [p.tcs.dt.year for p in yearly_periods]
        files = []
        for year in years:
            lookup_directory = self.cfg.get_lookup_dir(year)
            if lookup_directory is None:
                continue
            files.extend(sorted(lookup_directory.rglob(f"*{year}*.nc")))
        file_entries = [FDR4ALTFileNaming(f) for f in files]
        return pd.DataFrame(file_entries)
