# -*- coding: utf-8 -*-

# # Information of getting an input file list
# input_handler:
#
#     module_name: pysiral.envisat.iotools
#     class_name: EnvisatSGDRNC
#
#     options:
#         local_machine_def_tag: sgdr_v3p0  # -> l1b_repository.$platform.$input_tag in (local_machine_def.yaml)
#         lookup_dir: null  # Leave emtpy: This will be automatically filled with information from local_machine_def
#         # Example filename: ENV_RA_2_MWS____20021001T000511_20021001T005529_20170619T163625_3018_010_0004____PAC_R_NT_003.nc
#         filename_search: ENV_RA_2_MWS____{year:04d}{month:02d}{day:02d}T*.nc
#         filename_sep: _   # character to split the filename
#         tcs_str_index: 5  # index of time_coverage_start string in file (when splitting filename with filename_sep)
#
# # Class that will generate the initial l1p data object from
# # the input data
# input_adapter:
#
#     module_name: pysiral.envisat.l1_adapter
#     class_name: EnvisatSGDRNC
#
#     options:
#
#         name: "Envisat RA2/MWR Level 2 sensor geophysical data record (v3.0)"
#
#         # Radar parameters
#         bin_width_meter: 0.4686
#         radar_mode: lrm
#
#         # SGDR timestamp units
#         sgdr_timestamp_units: seconds since 2000-01-01 00:00:00.0
#         sgdr_timestamp_calendar: gregorian
#
#         # The timeliness is fixed (always reprocessed)
#         timeliness: rep
#
#         # expression to identify 1Hz parameters (re.search(variable_identifier_1Hz, variable_name))
#         # -> will be used to automatically interpolate range corrections from 1Hz to 20Hz, as these
#         #    are mixed in the Envisat SGDR
#         variable_identifier_1Hz: "01"
#
#         range_correction_targets:
#             dry_troposphere: mod_dry_tropo_cor_reanalysis_20
#             wet_troposphere: mod_wet_tropo_cor_reanalysis_20
#             inverse_barometric: inv_bar_cor_01
#             dynamic_atmosphere: hf_fluct_cor_01
#             ionosphere: iono_cor_gim_01_ku
#             ocean_tide_elastic: ocean_tide_sol1_01
#             ocean_tide_long_period: ocean_tide_eq_01
#             ocean_loading_tide: load_tide_sol1_01
#             solid_earth_tide: solid_earth_tide_01
#             geocentric_polar_tide: pole_tide_01
#             total_geocentric_ocean_tide: Null
#
#         classifier_targets:
#             peakiness_sgdr: peakiness_20_ku
#             sigma0: sig0_sea_ice_20_ku
#             sigma0_ice1: sig0_ice1_20_ku
#             sigma0_ice2: sig0_ice2_20_ku
#             leading_edge_width_ice2: width_leading_edge_ice2_20_ku
#             elevation_ice1: elevation_ice1_20_ku
#             geographic_correction_ice: geo_cor_range_20_ku
#             chirp_band: chirp_band_20_ku
#             dist_coast: dist_coast_20
#             noise_power: noise_power_20
#             offset_tracking: offset_tracking_20
#             orb_alt_rate: orb_alt_rate_20
#             range_sea_ice: range_sea_ice_20_ku
#             retracking_sea_ice_qual: retracking_sea_ice_qual_20_ku
#             swh_ocean: swh_ocean_20_ku
#             ice1_range: range_ice1_20_ku
#             ice2_range: range_ice2_20_ku
#             sitrack_range: range_sea_ice_20_ku
#             ocean_range: range_ocean_20_ku
#             slope_first_trailing_edge_ice2: slope_first_trailing_edge_ice2_20_ku


from collections import deque
from pathlib import Path
from typing import List

from dateperiods import DatePeriod
from pysiral.l1 import SourceFileDiscovery


class EnvisatSGDRNC(
    SourceFileDiscovery,
    supported_source_datasets=["envisat_sgdr_esa_v3p0"]
):

    def __init__(self, cfg) -> None:
        """
        File discovery for Envisat SGDG netcdf files with yyyy/mm/dd subfolder structure
        :param cfg:
        """

        # Save config
        self.cfg = cfg
        # Init empty file lists
        self._reset_file_list()

    def query_period(self, period: "DatePeriod") -> List[Path]:
        """
        Query for Sentinel Level-2 files for a specific period.
        :param period: dateperiods.DatePeriod
        :return: sorted list of filenames
        """
        # Make sure file list are empty
        self._reset_file_list()
        self._query(period)
        return self.sorted_list

    def _reset_file_list(self) -> None:
        """ Resets the result of previous file searches """
        self._list = deque([])
        self._sorted_list = []

    def _query(self, period: "DatePeriod") -> None:
        """
        Searches for files in the given period and stores result in property _sorted_list
        :param period: dateperiods.DatePeriod
        :return: None
        """

        # Loop over all months in the period
        daily_periods = period.get_segments("day")
        for daily_period in daily_periods:

            year, month, day = daily_period.tcs.year, daily_period.tcs.month, daily_period.tcs.day

            # Search folder
            lookup_folder = self._get_lookup_folder(year, month, day)
            if not Path(lookup_folder).is_dir():
                continue

            # Query the daily folder
            filename_search = self.cfg.filename_search.format(year=year, month=month, day=day)
            sgdr_files = list(Path(lookup_folder).glob(filename_search))

            # Add files to result list
            if not sgdr_files:
                continue
            self._sorted_list.extend(sorted(sgdr_files))

    def _get_lookup_folder(self, year, month, day) -> Path:
        yyyy, mm, dd = "%04g" % year, "%02g" % month, "%02g" % day
        return Path(self.cfg.lookup_dir) / yyyy / mm / dd

    @property
    def sorted_list(self) -> List[str]:
        return list(self._sorted_list)
