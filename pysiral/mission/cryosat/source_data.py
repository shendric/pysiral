# -*- coding: utf-8 -*-

import re
import numpy as np
import xarray
from collections import deque
from astropy.time import Time
from cftime import num2pydate
from dateperiods import DatePeriod

from loguru import logger
from scipy import interpolate
from typing import Optional, Dict, Tuple, List, Literal, Union
from pydantic import BaseModel, Field
from pathlib import Path

from pysiral import __version__ as pysiral_version
from pysiral.core.clocks import debug_timer
from pysiral.core.flags import ESA_SURFACE_TYPE_DICT
from pysiral.core.helper import parse_datetime_str
from pysiral.mission.cryosat import cs2_procstage2timeliness
from pysiral.l1 import Level1bData, SourceFileDiscovery, SourceDataLoader
from pysiral.waveform import CS2OCOGParameter


class ESACryoSat2ICEL1bFileConfig(BaseModel):
    """
    Default settings for the ESA CryoSat-2 ICE Product L1b data
    source data loader.
    """

    exclude_predicted_orbits: bool = False

    range_correction_targets: Dict[str, str] = {
        "dry_troposphere": "mod_dry_tropo_cor_01",
        "wet_troposphere": "mod_wet_tropo_cor_01",
        "inverse_barometric": "inv_bar_cor_01",
        "dynamic_atmosphere": "hf_fluct_total_cor_01",
        "ionosphere": "iono_cor_01",
        "ionosphere_gim": "iono_cor_gim_01",
        "ocean_tide_elastic": "ocean_tide_01",
        "ocean_tide_long_period": "ocean_tide_eq_01",
        "ocean_loading_tide": "load_tide_01",
        "solid_earth_tide": "solid_earth_tide_01",
        "geocentric_polar_tide": "pole_tide_01",
    }

    classifier_targets: Dict[str, str] = {
        "stack_peakiness": "stack_peakiness_20_ku",
        "stack_skewness": "stack_skewness_20_ku",
        "stack_scaled_amplitude": "stack_scaled_amplitude_20_ku",
        "stack_standard_deviation": "stack_std_20_ku",
        "stack_kurtosis": "stack_kurtosis_20_ku",
        "stack_centre": "stack_centre_20_ku",
        "stack_centre_angle": "stack_centre_angle_20_ku",
        "stack_centre_look_angle": "stack_centre_look_angle_20_ku",
        "transmit_power": "transmit_pwr_20_ku",
        "noise_power": "noise_power_20_ku",
        "look_angle_start": "look_angle_start_20_ku",
        "look_angle_stop": "look_angle_stop_20_ku",
        "stack_beams": "stack_number_after_weighting_20_ku",
        "uso_cor": "uso_cor_20_ku",
        "window_delay": "window_del_20_ku"
    }


class ESACryoSat2ICEL1bFile(
    SourceDataLoader,
    supported_source_datasets=[
        "cryosat2_rep_esa_iceb0E",
        "cryosat2_nrt_esa_iceb0E",
    ]
):
    """
    Source data loader class for ESA CryoSat-2 ICE Product L1b data

    :param raise_on_error: Flag determining the behaviour when the netCDF file
        cannot be read or converted into an L1 object.
    :param configuration_kwargs:
    """

    def __init__(
            self,
            raise_on_error: bool = False,
            **configuration_kwargs
    ) -> None:
        """
        Initializes the class
        """
        self.cfg = ESACryoSat2ICEL1bFileConfig(**configuration_kwargs)
        self.raise_on_error = raise_on_error

    @staticmethod
    def translate_opmode2radar_mode(op_mode: str) -> str:
        """
        Converts the ESA operation mode str in the pysiral compliant version

        :param op_mode: ESA style radar operation modes

        :return: pysiral style radar operation mode
        """
        translate_dict = {"sar": "sar", "lrm": "lrm", "sarin": "sin"}
        return translate_dict.get(op_mode)

    @debug_timer("Parsed CryoSat-2 Level-1b file")
    def get_l1(
            self,
            filepath: Path,
            polar_ocean_check=None
    ) -> Optional[Level1bData]:
        """
        Main entry point to the CryoSat-2 ICE Level-1b source data loader.

        :param filepath: The path of the source data file
        :param polar_ocean_check:

        :return: Level1bData instance or None (if error or polar_ocean_check has not passed)
        """

        # Create an empty Level-1 data object
        l1 = Level1bData()

        # Parse the input file
        nc = self._read_input_netcdf(filepath)
        if nc is None:
            return None

        # Legacy check
        if self._reject_predicted_orbit(nc):
            logger.debug("- reject data with predicted orbit")
            return None

        # Set metadata
        self._set_input_file_metadata(filepath, nc, l1)
        if polar_ocean_check is not None:
            has_polar_ocean_data = polar_ocean_check(l1.info)
            if not has_polar_ocean_data:
                logger.debug(f"- {has_polar_ocean_data=}")
                return None

        # Polar ocean check passed, now fill the rest with the l1 data groups
        self._set_l1_data_groups(nc, l1)

        # Return the l1 object
        return l1

    @staticmethod
    def get_wfm_range(window_delay: np.ndarray, n_range_bins: int) -> np.ndarray:
        """
        Returns the range for each waveform bin based on the window delay
        and the number of range bins

        :param window_delay: The two-way delay to the center of the range window in seconds
        :param n_range_bins: The number of range bins (256: sar, 512: sin)
        :return: The range for each waveform bin as array (time, ns)
        """
        lightspeed = 299792458.0
        bandwidth = 320000000.0
        # The two-way delay time give the distance to the central bin
        central_window_range = window_delay * lightspeed / 2.0
        # Calculate the offset from the center to the first range bin
        window_size = (n_range_bins * lightspeed) / (4.0 * bandwidth)
        first_bin_offset = window_size / 2.0
        # Calculate the range increment for each bin
        range_increment = np.arange(n_range_bins) * lightspeed / (4.0 * bandwidth)

        # Reshape the arrays
        range_offset = np.tile(range_increment, (window_delay.shape[0], 1)) - first_bin_offset
        window_range = np.tile(central_window_range, (n_range_bins, 1)).transpose()

        return window_range + range_offset

    @staticmethod
    def interp_1hz_to_20hz(
            variable_1hz: np.ndarray,
            time_1hz: np.ndarray,
            time_20hz: np.ndarray,
            **kwargs
    ) -> Tuple[np.ndarray, bool]:
        """
        Computes a simple linear interpolation to transform a 1Hz into a 20Hz variable

        :param variable_1hz: an 1Hz variable array
        :param time_1hz: 1Hz reference time
        :param time_20hz: 20 Hz reference time

        :return: the interpolated 20Hz variable
        """
        error_status = False
        try:
            f = interpolate.interp1d(time_1hz, variable_1hz, bounds_error=False, **kwargs)
            variable_20hz = f(time_20hz)
        except ValueError:
            fill_value = np.nan
            variable_20hz = np.full(time_20hz.shape, fill_value)
            error_status = True
        return variable_20hz, error_status

    def _read_input_netcdf(self, filepath: Path) -> Optional[xarray.Dataset]:
        """
        Read the netCDF file via xarray and returns the xarray.Dataset
        instance.

        :param filepath: The filepath to the L1b file

        :raise IOError: Any exception raised by xarray.load_dataset and
            raise_on_error=True
        """

        # Input Validation
        # NOTE: This is an error that always should be raised
        if not filepath.is_file():
            raise ValueError(f"Not a valid file: {filepath=}")

        # Read the netCDF file. The handling of any error within xarray
        # depends on the state of `raise_on_error` flag
        try:
            nc = xarray.load_dataset(filepath, decode_times=False, mask_and_scale=True)
        except Exception as e:
            nc = None
            msg = f"Error encountered by xarray {filepath=}"
            logger.error(msg)
            if self.raise_on_error:
                raise IOError(msg) from e
        return nc

    def _reject_predicted_orbit(self, nc: xarray.Dataset) -> bool:
        """
        Perform legacy check:

        An issue has been identified with baseline-D L1b data when the orbit solution
        is based on predicted orbits and not the DORIS solution (Nov 2020).
        The source of the orbit data can be identified by the `vector_source` global attribute
        in the L1b source files. This can take/should take the following values:

                nrt:  "fos predicted" (predicted orbit)
                      "doris_navigator" (DORIS Nav solution)
                rep:  "doris_precise" (final and precise DORIS solution)

        To prevent l1 data with erroneous orbit solution entering the processing chain, l1 data
        with the predicted orbit can be excluded here. The process of exclusion requires to set
        a flag in the l1 processor definition for the input handler:
            exclude_predicted_orbits: True

        :param nc:

        :return: Flag whether to reject particular file
        """

        exclude_predicted_orbits = self.cfg.exclude_predicted_orbits
        is_predicted_orbit = nc.vector_source.lower().strip() == "fos predicted"
        if is_predicted_orbit and exclude_predicted_orbits:
            logger.warning("Predicted orbit solution detected -> skip file")
            return True
        return False

    @staticmethod
    def _set_input_file_metadata(
            filepath: Path,
            nc: xarray.Dataset,
            l1: Level1bData
    ) -> None:
        """
        Populate the Level-1 data metadata with global attributes
        of the netCDF file.

        :param l1:
        :return:
        """

        metadata = nc.attrs
        info = l1.info

        # Processing environment metadata
        info.set_attribute("pysiral_version", pysiral_version)

        # General product metadata
        info.set_attribute("mission", "cryosat2")
        info.set_attribute("mission_sensor", "siral")
        # TODO: Baseline is currently hard-coded
        info.set_attribute("mission_data_version", "E")
        info.set_attribute("orbit", metadata["abs_orbit_start"])
        info.set_attribute("rel_orbit", metadata["rel_orbit_number"])
        info.set_attribute("cycle", metadata["cycle_number"])
        info.set_attribute("mission_data_source", filepath.name)
        info.set_attribute("timeliness", cs2_procstage2timeliness(metadata["processing_stage"]))

        # Time-Orbit Metadata
        tcs_tai = parse_datetime_str(metadata["first_record_time"][4:])
        tce_tai = parse_datetime_str(metadata["last_record_time"][4:])
        tcs_utc, tce_utc = Time([tcs_tai, tce_tai], scale="tai").utc.datetime

        lats = [float(metadata["first_record_lat"])*1e-6, float(metadata["last_record_lat"])*1e-6]
        lons = [float(metadata["first_record_lon"])*1e-6, float(metadata["last_record_lon"])*1e-6]
        info.set_attribute("start_time", tcs_utc)
        info.set_attribute("stop_time", tce_utc)
        info.set_attribute("lat_min", np.amin(lats))
        info.set_attribute("lat_max", np.amax(lats))
        info.set_attribute("lon_min", np.amin(lons))
        info.set_attribute("lon_max", np.amax(lons))

        # Product Content Metadata
        for mode in ["sar", "sin", "lrm"]:
            percent_value = 0.0
            if metadata["sir_op_mode"].strip().lower() == mode:
                percent_value = 100.
            info.set_attribute(f"{mode}_mode_percent", percent_value)
        info.set_attribute("open_ocean_percent", float(metadata["open_ocean_percent"])*0.01)

    def _set_l1_data_groups(self, nc: xarray.Dataset, l1: Level1bData) -> None:
        """
        Fill all data groups of the Level-1 data object with the content of the netCDF file.
        This is just the overview method, see specific sub-methods below

        :param nc: Content of the netCDF file
        :param l1: Level1bData instance

        :return: None, Level1bData is changed in place
        """
        self._set_time_orbit_data_group(nc, l1)
        self._set_waveform_data_group(nc, l1)
        self._set_range_correction_group(nc, l1)
        self._set_surface_type_group(nc, l1)
        self._set_classifier_group(nc, l1)

    @staticmethod
    def _set_time_orbit_data_group(nc: xarray.Dataset, l1: Level1bData) -> None:
        """
        Transfer the time orbit parameter from the netcdf to l1 data object
        :return: None
        """

        # Transfer the timestamp
        # NOTE: Here it is critical that the xarray does not automatically decode time since it is
        #       difficult to work with the numpy datetime64 date format. Better to compute datetimes using
        #       a know num2pydate conversion
        tai_datetime = num2pydate(nc.time_20_ku.values, units=nc.time_20_ku.units)
        l1.time_orbit.timestamp = Time(tai_datetime, scale="tai").utc.datetime

        # Set the geolocation
        l1.time_orbit.set_position(
            nc.lon_20_ku.values,
            nc.lat_20_ku.values,
            nc.alt_20_ku.values,
            nc.orb_alt_rate_20_ku.values)

        # Set antenna attitude
        l1.time_orbit.set_antenna_attitude(
            nc.off_nadir_pitch_angle_str_20_ku.values,
            nc.off_nadir_roll_angle_str_20_ku.values,
            nc.off_nadir_yaw_angle_str_20_ku.values)

    def _set_waveform_data_group(self, nc: xarray.Dataset, l1: Level1bData) -> None:
        """
        Transfer of the waveform group to the Level-1 object. This includes
          1. the computation of waveform power in Watts
          2. the computation of the window delay in meter for each waveform bin
          3. extraction of the waveform valid flag
        :return: None
        """

        # Get the waveform
        # NOTE: Convert the waveform units to Watts. From the documentation:is applied as follows:
        #       pwr_waveform_20_ku(time, ns) * echo_scale_factor_20_ku(time, ns) * 2 ^ echo_scale_pwr_20_ku(time)
        wfm_linear = nc.pwr_waveform_20_ku.values

        # Get the shape of the waveform array
        dim_time, dim_ns = wfm_linear.shape

        # Scaling parameter are 1D -> Replicate to same shape as waveform array
        echo_scale_factor = nc.echo_scale_factor_20_ku.values
        echo_scale_pwr = nc.echo_scale_pwr_20_ku.values
        echo_scale_factor = np.tile(echo_scale_factor, (dim_ns, 1)).transpose()
        echo_scale_pwr = np.tile(echo_scale_pwr, (dim_ns, 1)).transpose()

        # Convert the waveform from linear counts to Watts
        wfm_power = wfm_linear*echo_scale_factor * 2.0**echo_scale_pwr

        # Get the window delay
        # From the documentation:
        #   Calibrated 2-way window delay: distance from CoM to middle range window (at sample ns/2 from 0).
        #   It includes all the range corrections given in the variable instr_cor_range and in the
        #   variable uso_cor_20_ku. This is a 2-way time and 2-way corrections are applied.
        window_delay = nc.window_del_20_ku.values

        # Convert window delay to range for each waveform range bin
        wfm_range = self.get_wfm_range(window_delay, dim_ns)

        # Set the waveform
        op_mode = str(nc.attrs["sir_op_mode"].strip().lower())
        radar_mode = self.translate_opmode2radar_mode(op_mode)
        l1.waveform.set_waveform_data(wfm_power, wfm_range, radar_mode)

        # --- Get the valid flag ---
        #
        # From the documentation
        # :comment = "Measurement confidence flags. Generally the MCD flags indicate problems when set.
        #             If the whole MCD is 0 then no problems or non-nominal conditions were detected.
        #             Serious errors are indicated by setting the most significant bit, i.e. block_degraded,
        #             in which case the block must not be processed. Other error settings can be regarded
        #             as warnings.";
        #
        # :flag_masks = -2147483648, block_degraded        <- most severe error
        #                1073741824, blank_block
        #                536870912, datation_degraded
        #                268435456, orbit_prop_error
        #                134217728, orbit_file_change
        #                67108864, orbit_gap
        #                33554432, echo_saturated
        #                16777216, other_echo_error
        #                8388608, sarin_rx1_error
        #                4194304, sarin_rx2_error
        #                2097152, window_delay_error
        #                1048576, agc_error
        #                524288, cal1_missing
        #                262144, cal1_default
        #                131072, doris_uso_missing
        #                65536, ccal1_default
        #                32768, trk_echo_error
        #                16384, echo_rx1_error
        #                8192, echo_rx2_error
        #                4096, npm_error                   <- Defined as maximum permissible error level
        #                2048, cal1_pwr_corr_type
        #                128, phase_pert_cor_missing       <- Seems to be always set for SARin
        #                64, cal2_missing
        #                32, cal2_default
        #                16, power_scale_error
        #                8, attitude_cor_missing
        #                1, phase_pert_cor_default
        measurement_confident_flag = nc.flag_mcd_20_ku.values
        valid_flag = (measurement_confident_flag >= 0) & (measurement_confident_flag <= 4096)
        l1.waveform.set_valid_flag(valid_flag)

    def _set_range_correction_group(self, nc: xarray.Dataset, l1: Level1bData) -> None:
        """
        Transfer the range corrections defined in the l1p config file to the Level-1 object
        NOTE: The range corrections are all in 1 Hz and must be interpolated to 20Hz

        :return: None
        """

        # Get the reference times for interpolating the range corrections from 1Hz -> 20Hz
        time_1hz = nc.time_cor_01.values
        time_20hz = nc.time_20_ku.values

        # Loop over all range correction variables defined in the processor definition file
        for key in self.cfg.range_correction_targets.keys():
            pds_var_name = self.cfg.range_correction_targets[key]
            variable_1hz = getattr(nc, pds_var_name)
            variable_20hz, error_status = self.interp_1hz_to_20hz(variable_1hz.values, time_1hz, time_20hz)
            if error_status:
                msg = f"- Error in 20Hz interpolation for variable `{pds_var_name}` -> set only dummy"
                logger.warning(msg)
            l1.correction.set_parameter(key, variable_20hz)

    def _set_surface_type_group(self, nc: xarray.Dataset, l1: Level1bData) -> None:
        """
        Transfer of the surface type flag to the Level-1 object
        NOTE: In the current state (TEST dataset), the surface type flag is only 1 Hz. A nearest neighbour
              interpolation is used to get the 20Hz surface type flag.
        :return: None
        """

        # Get the reference times for interpolating the flag from 1Hz -> 20Hz
        time_1hz = nc.time_cor_01.values
        time_20hz = nc.time_20_ku.values

        # Interpolate 1Hz surface type flag to 20 Hz
        surface_type_1hz = nc.surf_type_01.values
        surface_type_20hz, error_status = self.interp_1hz_to_20hz(
            surface_type_1hz, time_1hz, time_20hz, kind="nearest"
        )
        if error_status:
            msg = "- Error in 20Hz interpolation for variable `surf_type_01` -> set only dummy"
            logger.warning(msg)

        # Set the flag
        for key in ESA_SURFACE_TYPE_DICT.keys():
            flag = surface_type_20hz == ESA_SURFACE_TYPE_DICT[key]
            l1.surface_type.add_flag(flag, key)

    def _set_classifier_group(self, nc: xarray.Dataset, l1: Level1bData) -> None:
        """
        Transfer the classifiers defined in the l1p config file to the Level-1 object.
        NOTE: It is assumed that all classifiers are 20Hz

        In addition, a few legacy parameter are computed based on the waveform counts that is only available at
        this stage. Computation of other parameter such as sigma_0, leading_edge_width, ... are moved to the
        post-processing

        :return: None
        """
        # Loop over all classifier variables defined in the processor definition file
        for key in self.cfg.classifier_targets.keys():
            variable_20hz = getattr(nc, self.cfg.classifier_targets[key])
            l1.classifier.add(variable_20hz, key)

        # Calculate the OCOG Parameter (CryoSat-2 notation)
        ocog = CS2OCOGParameter(l1.waveform.power)
        l1.classifier.add(ocog.width, "ocog_width")
        l1.classifier.add(ocog.amplitude, "ocog_amplitude")

        # Get satellite velocity vector (classifier needs to be vector -> manual extraction needed)
        satellite_velocity_vector = nc.sat_vel_vec_20_ku.values
        l1.classifier.add(satellite_velocity_vector[:, 0], "satellite_velocity_x")
        l1.classifier.add(satellite_velocity_vector[:, 1], "satellite_velocity_y")
        l1.classifier.add(satellite_velocity_vector[:, 2], "satellite_velocity_z")

# class ESACryoSat2PDSBaselineDPatchFES(ESACryoSat2PDSBaselineD):
#     def __init__(self, cfg, raise_on_error=False):
#         ESACryoSat2PDSBaselineD.__init__(self, cfg, raise_on_error)
#
#     def _set_l1_data_groups(self):
#         ESACryoSat2PDSBaselineD._set_l1_data_groups(self)
#         fespath = self._get_fes_path(self.filepath)
#         if not Path(fespath).is_file():
#             msg = f"Not a valid file: {fespath}"
#             logger.warning(msg)
#             self.error.add_error("invalid-filepath", msg)
#             raise FileNotFoundError
#         try:
#             nc_fes = xarray.open_dataset(fespath, decode_times=False, mask_and_scale=True)
#
#             # time_1hz = self.nc.time_cor_01.values
#             # time_20hz = self.nc.time_20_ku.values
#
#             msg = f"Patching FES2014b tide data from: {fespath}"
#             logger.info(msg)
#
#             # ocean_tide_elastic: ocean_tide_01
#             variable_20hz = getattr(nc_fes, 'ocean_tide_20')
#             # variable_20hz, error_status = self.interp_1hz_to_20hz(variable_1hz.values, time_1hz, time_20hz)
#             # if error_status:
#             #    msg = "- Error in 20Hz interpolation for variable `%s` -> set only dummy" % 'ocean_tide_01'
#             #    logger.warning(msg)
#             #    raise FileNotFoundError
#             self.l1.correction.set_parameter('ocean_tide_elastic', variable_20hz)
#
#             # ocean_tide_long_period: ocean_tide_eq_01
#             variable_20hz = getattr(nc_fes, 'ocean_tide_eq_20')
#             # variable_20hz, error_status = self.interp_1hz_to_20hz(variable_1hz.values, time_1hz, time_20hz)
#             # if error_status:
#             #    msg = "- Error in 20Hz interpolation for variable `%s` -> set only dummy" % 'ocean_tide_eq_01'
#             #    logger.warning(msg)
#             #    raise FileNotFoundError
#             self.l1.correction.set_parameter('ocean_tide_long_period', variable_20hz)
#
#             # ocean_loading_tide: load_tide_01
#             variable_20hz = getattr(nc_fes, 'load_tide_20')
#             # variable_20hz, error_status = self.interp_1hz_to_20hz(variable_1hz.values, time_1hz, time_20hz)
#             # if error_status:
#             #     msg = "- Error in 20Hz interpolation for variable `%s` -> set only dummy" % 'load_tide_01'
#             #     logger.warning(msg)
#             #     raise FileNotFoundError
#             self.l1.correction.set_parameter('ocean_loading_tide', variable_20hz)
#         except:
#             msg = f"Error encountered by xarray parsing: {fespath}"
#             self.error.add_error("xarray-parse-error", msg)
#             self.nc = None
#             logger.warning(msg)
#             raise FileNotFoundError
#
#     def _get_fes_path(self, filepath):
#         # TODO: get the substitutions to make from config file. Get a list of pairs of sub 'this' to 'that'.
#         # pathsubs = [ ( 'L1B', 'L1B/FES2014' ), ( 'nc', 'fes2014b.nc' ) ]
#         newpath = str(filepath)
#         p = re.compile('L1B')
#         newpath = p.sub('L1B/FES2014', newpath)
#         p = re.compile('nc')
#         newpath = p.sub('fes2014b.nc', newpath)
#         p = re.compile('TEST')
#         newpath = p.sub('LTA_', newpath)
#         return newpath


# class ESACryoSat2PDSBaselineDPatchFESArctide(ESACryoSat2PDSBaselineDPatchFES):
#     def __init__(self, cfg, raise_on_error=False):
#         ESACryoSat2PDSBaselineDPatchFES.__init__(self, cfg, raise_on_error)
#
#     def _set_l1_data_groups(self):
#         ESACryoSat2PDSBaselineDPatchFES._set_l1_data_groups(self)
#         arcpath = self._get_arctide_path(self.filepath)
#         if not Path(arcpath).is_file():
#             msg = f"Not a valid file: {arcpath}"
#             logger.warning(msg)
#             self.error.add_error("invalid-filepath", msg)
#             # The handling of missing files here is different so that we can still process
#             # south files even though we don't have Arctide for them
#             self.l1.correction.set_parameter('ocean_tide_elastic_2',
#                                              self.l1.correction.get_parameter_by_name('ocean_tide_elastic'))
#         else:
#             nc_arc = xarray.open_dataset(arcpath, decode_times=False, mask_and_scale=True)
#
#             # time_1hz = self.nc.time_cor_01.values
#             # time_20hz = self.nc.time_20_ku.values
#
#             msg = f"Patching ARCTIDE tide data from: {arcpath}"
#             logger.info(msg)
#
#             # ocean_tide_elastic: ocean_tide_01
#             variable_20hz = getattr(nc_arc, 'tide_Arctic')
#             # variable_20hz, error_status = self.interp_1hz_to_20hz(variable_1hz.values, time_1hz, time_20hz)
#             # if error_status:
#             #    msg = "- Error in 20Hz interpolation for variable `%s` -> set only dummy" % 'ocean_tide_01'
#             #    logger.warning(msg)
#             #    raise FileNotFoundError
#             nans_indices = np.where(np.isnan(variable_20hz))[0]
#             if len(nans_indices) > 0:
#                 msg = 'Arctide file had {numnan} NaN values of {numval}. These have been replaced with FES2014b data'.format(numnan=len(nans_indices), numval=len(variable_20hz))
#                 logger.warning(msg)
#                 variable_20hz[nans_indices] = self.l1.correction.get_parameter_by_name('ocean_tide_elastic')[nans_indices].values
#             self.l1.correction.set_parameter('ocean_tide_elastic_2', self.l1.correction.get_parameter_by_name('ocean_tide_elastic'))
#             self.l1.correction.set_parameter('ocean_tide_elastic', variable_20hz)
#
#
#     def _get_arctide_path(self, filepath):
#         # TODO: get the substitutions to make from config file. Get a list of pairs of sub 'this' to 'that'.
#         # pathsubs = [ ( 'L1B', 'L1B/FES2014' ), ( 'nc', 'fes2014b.nc' ) ]
#         newpath = str(filepath)
#         p = re.compile('L1B')
#         newpath = p.sub('L1B/ARCTIDE', newpath)
#         p = re.compile('nc')
#         newpath = p.sub('RegAT_Arctic_tides_v1.2.nc', newpath)
#         p = re.compile('TEST')
#         newpath = p.sub('LTA_', newpath)
#         return newpath


# class ESACryoSat2PDSBaselineDPatchFESArctideDiscrim(ESACryoSat2PDSBaselineDPatchFESArctide):
#     def __init__(self, cfg, raise_on_error=False):
#         ESACryoSat2PDSBaselineDPatchFESArctide.__init__(self, cfg, raise_on_error)
#
#     def _set_l1_data_groups(self):
#         ESACryoSat2PDSBaselineDPatchFESArctide._set_l1_data_groups(self)
#         discpath = self._get_disc_path(self.filepath)
#         if not Path(discpath).is_file():
#             msg = f"Not a valid file: {discpath}"
#             logger.warning(msg)
#             self.error.add_error("invalid-filepath", msg)
#             raise FileNotFoundError
#         else:
#             nc_arc = xarray.open_dataset(discpath, decode_times=False, mask_and_scale=True)
#
#             # time_1hz = self.nc.time_cor_01.values
#             # time_20hz = self.nc.time_20_ku.values
#
#             msg = f"Patching discrimination data from: {discpath}"
#             logger.info(msg)
#
#             variable_20hz = getattr(nc_arc, 'class')
#
#             nans_indices = np.where(np.isnan(variable_20hz))[0]
#             if len(nans_indices) > 0:
#                 msg = 'Discrimination data has NaN values'
#                 logger.warning(msg)
#                 variable_20hz[nans_indices] = -1
#
#             # The IW ATBD says 2, 4, 6 are leads and 1, 10 are sea ice
#             self.l1.classifier.add(variable_20hz.astype(int), 'cls_nn_discrimination')
#
#
#     def _get_disc_path(self, filepath):
#         # TODO: get the substitutions to make from config file. Get a list of pairs of sub 'this' to 'that'.
#         # pathsubs = [ ( 'L1B', 'L1B/DISCRIM' ), ( 'nc', 'fes2014b.nc' ) ]
#         newpath = str(filepath)
#         p = re.compile('L1B')
#         newpath = p.sub('L1B/DISCRIM', newpath)
#         p = re.compile('.nc')
#         newpath = p.sub('_class.nc', newpath)
#         p = re.compile('TEST')
#         newpath = p.sub('LTA_', newpath)
#         return newpath

class CS2ICEFileDiscoveryConfig(BaseModel):
    """
    Configuration for ESA CryoSat-2 ICE File discovery.
    (Default values for baseline-E).

    """
    lookup_modes: List[Literal["sar", "sin", "lrm"]] = Field(
        default=["sar", "sin", "lrm"],
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
    SourceFileDiscovery,
    supported_source_datasets=[
        "cryosat2_rep_esa_iceb0E",
        "cryosat2_nrt_esa_iceb0E"
    ]
):
    """
    Class for file discovery of CryoSat-2 ICE Level-1b data products (SAR, SIN, LRM).

    The class expects as in
    """

    def __init__(
            self,
            lookup_directory: Dict[str, Path],
            **options_kwargs: Dict
    ) -> None:
        """
        Initialize the

        :param lookup_directory:
        :param options_kwargs:
        """
        self._lookup_directory = lookup_directory
        self.cfg = CS2ICEFileDiscoveryConfig(**options_kwargs)

    def query_period(self, period: DatePeriod) -> List[Path]:
        """
        Return a list of sorted files

        :param period:
        :return:
        """
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
        if not lookup_dir.is_dir():
            logger.error(f"{lookup_dir=} does not exist")
            return
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
        logger.info(f"Found {n_files} {mode} files")

    def _get_files_per_day(self, lookup_dir, year, month, day):
        """ Return a list of files for a given lookup directory """
        # Search for specific day
        filename_search = self.cfg.filename_search.format(year=year, month=month, day=day)
        return sorted(Path(lookup_dir).glob(filename_search))

    def _get_lookup_dir(self, year, month, mode):
        yyyy, mm = "%04g" % year, "%02g" % month
        return Path(self.lookup_directory[mode]) / yyyy / mm

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
    def lookup_directory(self) -> Union[Path, Dict[str, Path]]:
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
