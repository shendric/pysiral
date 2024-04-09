# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks"

from pathlib import Path

import numpy as np
from cftime import num2pydate
from loguru import logger
from typing import Optional

from pysiral import psrlcfg
from pysiral.core.clocks import StopWatch
from pysiral.core.flags import ESA_SURFACE_TYPE_DICT, ORCondition
from pysiral.mission.ers.sgdrfile import ERSSGDR
from pysiral.l1 import SourceDataLoader, Level1bData

# # Information of getting an input file list
# input_handler:
#
#     module_name: pysiral.ers.iotools
#     class_name: ERSCycleBasedSGDR
#
#     options:
#         local_machine_def_tag: sgdr_reaper_v1p08  # -> l1b_repository.$platform.$input_tag in (local_machine_def.yaml)
#         lookup_dir: null  # Leave emtpy: This will be automatically filled with information from local_machine_def
#         cycle_folder_regex: E._REAP_ERS_ALT_2S_.
#         folder_parser: E{ers_number:1d}_REAP_ERS_ALT_2S_{tcs}_{tce}_{cycle:04d}
#         filename_search: E*_REAP_ERS_ALT_2S_{year:04d}{month:02d}{day:02d}T*_RP01.NC
#         filename_sep: _   # character to split the filename
#         tcs_str_index: 5  # index of time_coverage_start string in file (when splitting filename with filename_sep)

# input_adapter:
#
#     module_name: pysiral.ers.l1_adapter
#     class_name: ERSReaperSGDR
#
#     options:
#
#         name: "Radar Altimeter REAPER Sensor Geophysical Data Record - SGDR "
#
#         # Radar Altimeter Settings
#         range_bin_width: 0.4545
#         nominal_tracking_bin: 32.5
#
#         # SGDR Input Settings
#         sgdr_n_blocks: 20
#
#         # SGDR timestamp units
#         sgdr_timestamp_units: seconds since 1990-01-01 00:00:00
#         sgdr_timestamp_calendar: gregorian
#
#         platform_name_dict:
#             E1: ers1
#             E2: ers2
#
#
#         # The timeliness is fixed (always reprocessed)
#         timeliness: rep
#
#         range_correction_targets:
#             dry_troposphere: model_dry_tropo_corr
#             wet_troposphere: model_wet_tropo_corr
#             inverse_barometric: inv_bar_corr
#             dynamic_atmosphere: hf_fluctuations_corr
#             ionosphere: iono_corr_model
#             ocean_tide_elastic: ocean_tide_sol1
#             ocean_tide_long_period: ocean_tide_equil
#             ocean_loading_tide: load_tide_sol1
#             solid_earth_tide: solid_earth_tide
#             geocentric_polar_tide: pole_tide
#             total_geocentric_ocean_tide: Null
#
#         classifier_targets:
#             chirp_type: alt_state_flag_chirp_type_20hz
#             peakiness_sgdr: peakiness_20hz
#             ocog_width: width_20hz
#             ocog_amplitude: amplitude_20hz
#             sigma0: sitrack_sig0_20hz
#             sigma0_ice1: ice1_sig0_20hz
#             sigma0_ice2: ice2_sig0_20hz
#             slope: slope
#             leading_edge_width_ice2: ice2_le_width_20hz
#             ice1_range: ice1_range_20hz
#             ice2_range: ice2_range_20hz
#             sitrack_range: sitrack_range_20hz
#             ocean_range: ocean_range_20hz
#             significant_wave_height: swh_20hz
#             center_of_gravity_corr: cog_corr_20hz
#             epsilon_sec: htl_disc_out
#             alt_state_flag: alt_state_flag_20hz
#             alt_state_flag_chirp_type: alt_state_flag_chirp_type_20hz
#             alt_state_flag_tracking_type: alt_state_flag_tracking_type_20hz
#             alt_state_flag_tracking_surface: alt_state_flag_tracking_surface_20hz
#             noise_floor: noise_floor
#
#
# # Output handler (will always be default handler -> only options)
# output_handler:
#
#     options:
#         local_machine_def_tag: sgdr_reaper_v1p08
#         version:
#             source_file_tag: esa_sgdr_v1p08
#             version_file_tag: v1p3

class ERSReaperSGDR(
    SourceDataLoader,
    supported_source_datasets=[
        "ers1_sgdr_esa_v1p8",
        "ers2_sgdr_esa_v1p8"
    ]
):
    """ Converts a Envisat SGDR object into a L1bData object """

    def __init__(self, cfg, raise_on_error=False):
        """
        Input handler for Sentinel-3 L2WAT netCDF files from the CODA.
        :param cfg: A treedict object (root.input_handler.options) from the corresponding Level-1 pre-processor
                    config file
        :param raise_on_error: Boolean value if the class should raise an exception upon an error (default: False)
        """

        # Debug variables
        self.cfg = cfg
        self.raise_on_error = raise_on_error
        self.timer = None
        self.l1 = None
        self.filepath = None

    def get_l1(self, filepath: Path) -> Optional[Level1bData]:
        """
        Read the Envisat SGDR file and transfers its content to a Level1Data instance
        :param filepath: The full file path to the netCDF file
        :return: The parsed (or empty) Level-1 data container
        """

        # Store arguments
        self.filepath = filepath

        # Create an empty Level-1 data object
        self.l1 = Level1bData()

        #  for debug purposes
        self.timer = StopWatch()
        self.timer.start()

        # Read the file
        # NOTE: This will create the variable `self.sgdr`
        self._read_sgdr()

        # Get metadata
        self._set_input_file_metadata()

        # Polar ocean check passed, now fill the rest of the l1 data groups
        self._set_l1_data_groups()

        self.timer.stop()
        logger.info("- Created L1 object in %.3f seconds" % self.timer.get_seconds())

        return self.l1

    def _read_sgdr(self):
        """ Read the L1b file and create a ERS native L1b object """
        self.sgdr = ERSSGDR(self.cfg)
        self.sgdr.filename = self.filepath
        self.sgdr.parse()
        error_status = self.sgdr.get_status()
        if error_status:
            # TODO: Needs ErrorHandler
            raise IOError()
        self.sgdr.post_processing()

    def _set_input_file_metadata(self):
        """ Extract essential metadata information from SGDR file """
        info = self.l1.info
        sgdr = self.sgdr
        info.set_attribute("pysiral_version", psrlcfg.version)
        try:
            info.set_attribute("mission", self.cfg.platform_name_dict[str(sgdr.nc.mission)])
        except KeyError:
            mission_id = self.sgdr.guess_mission_from_filename()
            info.set_attribute("mission", self.cfg.platform_name_dict[str(mission_id)])

        info.set_attribute("mission_data_version", sgdr.nc.software_ver)
        info.set_attribute("orbit", sgdr.nc.abs_orbit)
        info.set_attribute("cycle", sgdr.nc.cycle)
        mission_data_source = Path(sgdr.nc.filename).name
        info.set_attribute("mission_data_source", mission_data_source)
        info.set_attribute("timeliness", self.cfg.timeliness)

    def _set_l1_data_groups(self):
        self._transfer_timeorbit()            # (lon, lat, alt, time)
        self._transfer_waveform_collection()  # (power, range)
        self._transfer_range_corrections()    # (range corrections)
        self._transfer_surface_type_data()    # (land flag, ocean flag, ...)
        self._transfer_classifiers()          # (beam parameters, flags, ...)

    def _transfer_timeorbit(self) -> None:
        """
        Extracts the time/orbit data group from the SGDR data and set
        the l1 timeorbit data group

        :raises None:

        :return: None
        """

        # Transfer the timestamp
        sgdr_timestamp = self.sgdr.nc.time_20hz.flatten()
        units = self.cfg.sgdr_timestamp_units
        calendar = self.cfg.sgdr_timestamp_calendar
        timestamp = num2pydate(sgdr_timestamp, units, calendar)
        self.l1.time_orbit.timestamp = timestamp

        # Get the 20Hz positions
        lon, lat, alt = (
            self.sgdr.nc.lon_20hz.flatten(),
            self.sgdr.nc.lat_20hz.flatten(),
            self.sgdr.nc.alt_20hz.flatten()
        )

        # Check for invalid positions
        # (anecdotal evidence of invalid latitude values in reaper files)
        lon_is_valid = np.logical_and(lon >= -180., lon <= 180)
        lat_is_valid = np.logical_and(lat >= -90., lat <= 90)
        is_valid = np.logical_and(lon_is_valid, lat_is_valid)
        if not is_valid.all():
            idxs = np.where(np.logical_not(is_valid))[0]
            logger.error(f"- Found {len(idxs)} invalid lat/lon positions -> Set to NaN")
            lon[idxs] = np.nan
            lat[idxs] = np.nan
            alt[idxs] = np.nan

        # breakpoint()

        # Transfer the orbit position
        self.l1.time_orbit.set_position(lon, lat, alt)

        # Mandatory antenna pointing parameter (but not available for ERS)
        dummy_angle = np.full(timestamp.shape, 0.0)
        mispointing_deg = np.rad2deg(self.sgdr.nc.off_nadir_angle_wf_20hz.flatten())
        self.l1.time_orbit.set_antenna_attitude(dummy_angle, dummy_angle, dummy_angle, mispointing=mispointing_deg)

        # Update meta data container
        self.l1.update_data_limit_attributes()

    def _transfer_waveform_collection(self):
        """ Transfers the waveform data (power & range for each range bin) """

        # Transfer the reformed 18Hz waveforms
        self.l1.waveform.set_waveform_data(
            self.sgdr.wfm_power,
            self.sgdr.wfm_range,
            self.sgdr.radar_mode)

        # Set valid flag to exclude calibration data
        # (see section 3.5 of Reaper handbook)
        tracking_state = self.sgdr.nc.alt_state_flag_20hz.flatten()
        valid = ORCondition()
        valid.add(tracking_state == 2)
        valid.add(tracking_state == 3)
        self.l1.waveform.set_valid_flag(valid.flag)

    def _transfer_range_corrections(self):
        """
        Transfer range correction data from the SGDR netCDF to the
        l1bdata object. The parameter are defined in
        config/mission_def.yaml for ers1/ers2
        -> ersX.settings.sgdr_range_correction_targets

        For a description of the parameter see section 3.10 in the
        REAPER handbook
        """
        grc_dict = self.cfg.range_correction_targets
        for name in grc_dict.keys():
            target_parameter = grc_dict[name]
            if target_parameter is None:
                continue
            correction = getattr(self.sgdr.nc, target_parameter)
            correction = np.repeat(correction, self.cfg.sgdr_n_blocks)
            self.l1.correction.set_parameter(name, correction)

    def _transfer_classifiers(self):
        """
        Transfer classifier parameter from the SGDR netCDF to the
        l1bdata object. Most parameter are defined in
        config/mission_def.yaml for ers1/ers2
        -> ersX.settings.sgdr_range_correction_targets
        """
        target_dict = self.cfg.classifier_targets
        for parameter_name in target_dict.keys():
            nc_parameter_name = target_dict[parameter_name]
            nc_parameter = getattr(self.sgdr.nc, nc_parameter_name)
            self.l1.classifier.add(nc_parameter.flatten(), parameter_name)

    def _transfer_surface_type_data(self):
        surface_type = self.sgdr.nc.surface_type
        surface_type = np.repeat(surface_type, self.cfg.sgdr_n_blocks)
        for key in ESA_SURFACE_TYPE_DICT.keys():
            flag = surface_type == ESA_SURFACE_TYPE_DICT[key]
            self.l1.surface_type.add_flag(flag, key)

    @property
    def empty(self):
        """
        Default return object, if nodata should be returned
        :return: Representation of an empty object (None)
        """
        return None
