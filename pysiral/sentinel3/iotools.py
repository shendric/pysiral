# -*- coding: utf-8 -*-

import os
from collections import deque
from datetime import datetime
from typing import List
from pathlib import Path

from dateperiods import DatePeriod
from loguru import logger
from parse import parse

from pysiral.core.clocks import debug_timer
from pysiral.l1preproc import SourceFileDiscovery


class L2SeaIceFileDiscovery(
    SourceFileDiscovery,
    supported_source_datasets=[
        "sentinel3a_rep_l2lansi_v5p0",
        "sentinel3a_nrt_l2lansi_v5p0",
        "sentinel3b_rep_l2lansi_v5p0",
        "sentinel3b_nrt_l2lansi_v5p0"
    ]
):
    """ Class to retrieve Sentinel-3 SRAL files from the Copernicus Online Data Archive """

    def __init__(self, cfg):
        """

        :param cfg: dict/treedict configuration options (see l1proc config file)
        """

        # Save config
        self.cfg = cfg

        # Create inventory
        logger.info(f"Sentinel-3 source directory: {cfg.lookup_dir}")
        self.catalogue = self._get_dataset_catalogue()
        logger.info(f"Found {self.n_catalogue_files} files ({self.cfg.filename_search})")

        # Init empty file lists
        self._reset_file_list()

    @debug_timer("S3 file catalog creation")
    def _get_dataset_catalogue(self):
        """
        Create a catalogues with the time coverage of the files on the server
        :return:
        """

        # Simple catalogue format
        # [(datetime, filepath), (datetime, filepath), ...]
        return [
            (nc_filepath, S3FileNaming(nc_filepath.parent.parts[-1]).tcs_dt)
            for nc_filepath in Path(self.cfg.lookup_dir).glob(f"**/{self.cfg.filename_search}")
        ]

    def get_file_for_period(self, period: DatePeriod) -> List[Path]:
        """
        Query for Sentinel Level-2 files for a specific period.

        :param period: dateperiods.DatePeriod

        :return: sorted list of filenames
        """
        # Make sure file list are empty
        self._reset_file_list()
        self._query(period)
        return self.sorted_list

    def _query(self, period: DatePeriod) -> None:
        """
        Searches for files in the given period and stores result in property _sorted_list
        :param period: dateperiods.DatePeriod
        :return: None
        """

        # Loop over all months in the period
        file_list = [filepath for filepath, dt in self.catalogue if period.tcs.dt <= dt <= period.tce.dt]
        self._sorted_list = sorted(file_list)

    def _reset_file_list(self):
        """ Resets the result of previous file searches """
        self._list = deque([])
        self._sorted_list = []

    @property
    def sorted_list(self):
        """ Return the search result """
        return self._sorted_list

    @property
    def n_catalogue_files(self) -> int:
        """
        Return the number of files in the file catalogue
        :return:
        """
        return len(self.catalogue)


def get_sentinel3_l1b_filelist(folder, target_nc_filename):
    """ Returns a list with measurement.nc files for given month """
    s3_file_list = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            if name == target_nc_filename:
                # Get the start datetime from the folder name
                datestr = os.path.split(root)[-1].split("_")[7]
                s3_file_list.append((os.path.join(root, name), datestr))
    return s3_file_list


class S3FileNaming(object):
    """
    Deciphering the Sentinel-3 filenaming convention
    (source: Sentinel 3 PDGS File Naming Convention (EUM/LEO-SEN3/SPE/10/0070, v1D, 24 June 2016)
    """

    def __init__(self, filename: str) -> None:
        """
        Decode the Sentinel-3 filename

        :param filename: The filename to be decoded
        """

        self.filenaming_convention = "{mission_id:3}_{data_source:2}_{processing_level:1}_" \
                                     "{data_type_id:6}_{time_coverage_start:15}_{time_coverage_end:15}_" \
                                     "{creation_time:15}_{instance_id:3}_{product_class_id:8}.{extension}"

        self.date_format = "%Y%m%dT%H%M%S"

        self.elements = parse(self.filenaming_convention, filename)
        if self.elements is None:
            raise ValueError(f"{filename} is not a valid sentinel3 filename [{self.filenaming_convention}")

    def _get_dt(self, start_time_str: str) -> datetime:
        """
        Convert a string to datetime
        :param start_time_str:
        :return:
        """
        return datetime.strptime(start_time_str, self.date_format)

    @property
    def dict(self) -> dict:
        """
        Dictionary of elements
        :return:
        """
        return dict(**self.elements.named)

    @property
    def tcs_dt(self) -> dict:
        """
        Dictionary of elements
        :return:
        """
        return self._get_dt(self.dict["time_coverage_start"])

    @property
    def tce_dt(self) -> dict:
        """
        Dictionary of elements
        :return:
        """
        return self._get_dt(self.dict["time_coverage_end"])
