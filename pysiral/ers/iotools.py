# -*- coding: utf-8 -*-


import os
import re
from typing import List
from collections import deque
from datetime import timedelta
from pathlib import Path

from dateperiods import DatePeriod
from dateutil import parser
from parse import compile

from pysiral.l1preproc import SourceFileDiscovery


class ERSCycleBasedSGDR(
    SourceFileDiscovery,
    supported_source_datasets=[
        "ers1_sgdr_esa_v1p8",
        "ers2_sgdr_esa_v1p8"
    ]
):

    def __init__(self, cfg):
        """
        File discovery for a cycle based order
        :param cfg:
        """

        # Save config
        self.cfg = cfg

        # Establish a lookup table that maps cycles to days from the cycle folder names
        self._create_date_lookup_table()

        # Init empty file lists
        self._reset_file_list()

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

    def _create_date_lookup_table(self):
        """
        Get a lookup table based on the folder names for each cycle. The content will be stored in
        self._date_lookup_table
        :return: None
        """

        # --- Parameters ---
        lookup_dir = self.cfg.lookup_dir  # The main repository directory
        regex = self.cfg.cycle_folder_regex  # The regex expression used to identify cycle folders
        folder_parser = compile(self.cfg.folder_parser)  # a parser string indicating parameters in folder name

        # --- Find all cycle folders ---
        # Note: the regex only applies to the sub-folder name, not the full path
        cycle_folders = [x[0] for x in os.walk(lookup_dir) if re.match(regex, os.path.split(x[0])[-1])]

        # --- Construct lookup table from each sub-folder name ---
        self._lookup_table = []
        for cycle_folder in cycle_folders:

            # Parse the folder name
            result = folder_parser.parse(os.path.split(cycle_folder)[-1])

            # Get start and end coverage as datetimes
            tcs, tce = parser.parse(result["tcs"]), parser.parse(result["tce"])

            # Compute list of dates between two datetimes
            delta = tce - tcs
            dates = []
            for i in range(delta.days + 1):
                date = tcs + timedelta(days=i)
                dates.append("%04g-%02g-%02g" % (date.year, date.month, date.day))

            # Add entry to lookup table
            result_dict = {item[0]: item[1] for item in result.named.items()}
            self._lookup_table.append(dict(dir=cycle_folder, dates=dates, **result_dict))

    def _reset_file_list(self):
        """ Resets the result of previous file searches """
        self._list = deque([])
        self._sorted_list = []

    def _query(self, period):
        """
        Searches for files in the given period and stores result in property _sorted_list
        :param period: dateperiods.DatePeriod
        :return: None
        """

        # Loop over all months in the period
        daily_periods = period.get_segments("day")
        for daily_period in daily_periods:

            year, month, day = daily_period.tcs.year, daily_period.tcs.month, daily_period.tcs.day

            # The date format in the lookup table is yyyy-mm-dd
            datestr = "%04g-%02g-%02g" % (year, month, day)

            # Get a list of cycle folders
            cycle_folders = self._get_cycle_folders_from_lookup_table(datestr)
            if len(cycle_folders) > 2:
                raise IOError(f"Date {datestr} in more than 2 cycle folders (this should not happen)")

            # Query each cycle folder
            filename_search = self.cfg.filename_search.format(year=year, month=month, day=day)
            for cycle_folder in cycle_folders:
                sgdr_files = Path(cycle_folder).glob(filename_search)
                self._sorted_list.extend(sorted(sgdr_files))

    def _get_cycle_folders_from_lookup_table(self, datestr):
        """
        Return a list of cycle folders that contain a date. Should be between 0 and 2
        :param datestr:
        :return:
        """
        return [entry["dir"] for entry in self._lookup_table if datestr in entry["dates"]]

    @property
    def sorted_list(self):
        return list(self._sorted_list)
