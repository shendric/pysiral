# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks"

import re
import xarray as xr
from pathlib import Path
from typing import Any, Dict, Union, Optional

import numpy as np
from cftime import num2pydate
from loguru import logger
from pydantic import BaseModel
from scipy import interpolate

from pysiral import PSRLCFG
from pysiral.core.clocks import StopWatch
from pysiral.core.flags import ESA_SURFACE_TYPE_DICT
from pysiral.core.iotools import ReadNC
from pysiral.core.legacy_classes import AttrDict
from pysiral.mission import SourceDataLoader
from pysiral.mission.envisat.functions import get_envisat_wfm_range
from pysiral.l1.l1_data import Level1bData
from pysiral.l1 import Level1PInputHandlerBase


class EnvisatFDR4ALTConfig(BaseModel):
    """
    Configuration for Envisat FDR4ALT data loader.
    """
    name: str
    classifier_targets: Dict[str, str]
    range_correction_targets: Dict[str, Optional[str]] = {}
    bin_width_meter: float = 0.4686
    radar_mode: str = "lrm"
    timeliness: str = "rep"
    raise_on_error: bool = True


class EnvisatFDR4ALT(SourceDataLoader):

    def __init__(self, **kwargs) -> None:
        """
        Input handler for Envisat FDR4ALT netCDF files combining the altimeter data from
        Fundamental Data Record (FDR) and the range correction (tides & atmospheric corrections)
        from the thematic sea ice product.

        :param cfg: Options from the corresponding Level-1 pre-processor config file
        :param raise_on_error: Boolean value if the class should raise an exception upon an error (default: False)
        """
        # Properties
        self.cfg = EnvisatFDR4ALTConfig(**kwargs)

    def get_l1(self, filepath: Path, **kwargs) -> Optional[Level1bData]:
        """
        Read the Envisat FDR4ALT file and transfers its content to a Level1Data instance

        :param filepath: The full file path to the netCDF file
        :return: The parsed (or empty) Level-1 data container
        """
        l1 = Level1bData()
        # NOTE: `decode_times=False` prevents a pandas.errors.OutOfBoundsTimedelta exception
        with xr.open_datatree(filepath, decode_times=False) as fdr_ds:
            self._set_input_file_metadata(l1, fdr_ds)
            self._transfer_data_groups(l1, fdr_ds)
        return l1

    def _set_input_file_metadata(self, l1: Level1bData, fdr_ds: xr.DataTree) -> None:
        """
        Extract dataset attributes from FDR dataset and store them
        in the Level-1 data object.

        :param l1: Level-1 data object to store the metadata
        :param fdr_ds: xarray DataTree object containing the FDR4ALT data

        :return: None (l1 is change in place)
        """
        l1.info.set_attribute("pysiral_version", PSRLCFG.version)
        l1.info.set_attribute("mission", "envisat")
        l1.info.set_attribute("mission_data_version", fdr_ds.attrs.get("source"))
        l1.info.set_attribute("orbit", fdr_ds.attrs.get("pass_number"))
        l1.info.set_attribute("cycle", fdr_ds.attrs.get("cycle_number"))
        l1.info.set_attribute("mission_data_source", fdr_ds.attrs.get("title"))
        l1.info.set_attribute("timeliness", self.cfg.timeliness)

    def _transfer_data_groups(self, *args) -> None:
        """
        Transfer data from the necessary data groups in the FDR dataset to the Level-1 data object.

        :param l1: Level-1 data object to store the transferred data
        :param fdr_ds: xarray DataTree object containing the FDR4ALT data

        :return: None (l1 is changed in place)
        """
        self._transfer_timeorbit(*args)
        self._transfer_waveform_collection(*args)
        self._transfer_range_corrections(*args)  # NOTE: Not included in the FDR4ALT product
        self._transfer_surface_type_data(*args)
        self._transfer_classifiers(*args)

    @staticmethod
    def _transfer_timeorbit(l1: Level1bData, fdr_ds: xr.DataTree) -> None:
        """
        Extracts the time/orbit data group from the FDR4ALT FDR data

        :param l1: Level-1 data object to store the transferred data
        :param fdr_ds: xarray DataTree object containing the FDR4ALT data

        :return: None (l1 is changed in place)
        """
        # and ensure longitude is -180:180 degrees east
        longitude_degrees_east = (fdr_ds.main.longitude.values + 180) % 360 - 180
        l1.time_orbit.set_position(
            longitude_degrees_east,
            fdr_ds.main.latitude.values,
            fdr_ds.main.altitude.values,
            fdr_ds.main.altitude_rate.values,
        )
        # Transfer the time
        fdr_time_var = fdr_ds.main.time
        timestamp = num2pydate(
            fdr_time_var.values,
            fdr_time_var.attrs["units"],
            fdr_time_var.attrs.get("calendar", "standard")
        )
        l1.time_orbit.timestamp = timestamp

        # Mandatory antenna pointing parameter (but not available for Envisat)
        dummy_angle = np.full(timestamp.shape, 0.0)
        l1.time_orbit.set_antenna_attitude(dummy_angle, dummy_angle, dummy_angle)

        # Update meta data container
        l1.update_data_limit_attributes()

    def _transfer_waveform_collection(self, l1: Level1bData, fdr_ds: xr.DataTree) -> None:

        # Transfer the reformed 18Hz waveforms
        # "waveform samples (I2+Q2, 1/2048 FFT power unit): 18 Hz Ku band";
        # "the echo is corrected for the intermediate frequency filter effect";
        wfm_power = fdr_ds.main.Ku.power_waveform.values
        n_records, n_range_bins = wfm_power.shape

        # From the FDR4ALT product handbook
        # 6.2.8 Offset tracking flag and reference Abscissa (ENVISAT)
        chirp_band_width_dict = {
            0: 320_000_000.,  # Ku band Ocean Mode
            1:  82_500_000,   # Lu band Ice Mode
            2:  20_000_000,   # ?
        }
        bandwidth = np.vectorize(chirp_band_width_dict.get)(fdr_ds.main.Ku.chirp_band.values)
        nominal_tracking_bin = 63 + fdr_ds.main.offset_tracking.values/256 * bandwidth / 320_000_000.0

        # Window delay is defined as the range to the start of the range window
        window_delay_m = fdr_ds.main.Ku.tracker_range.values - nominal_tracking_bin * self.cfg.bin_width_meter

        # Compute the range value for each range bin of the 18hz waveform
        wfm_range = get_envisat_wfm_range(
            window_delay_m,
            n_range_bins,
            bin_width_meter=self.cfg.bin_width_meter
        )
        # Transfer data to the waveform group
        l1.waveform.set_waveform_data(wfm_power, wfm_range, self.cfg.radar_mode)

        # Set valid flag to exclude calibration data
        all_flags = np.stack(
            (
                fdr_ds.main.mcd_flag_agc_fault_id.astype(bool),
                fdr_ds.main.mcd_flag_rx_delay_fault_id.astype(bool),
                fdr_ds.main.mcd_flag_proc_error.astype(bool),
                fdr_ds.main.mcd_flag_uso.values.astype(bool),
                fdr_ds.main.mcd_flag_waveform_fault_id.values.astype(bool),
            )
        )
        valid_flag = np.logical_not(np.max(all_flags, axis=0))
        l1.waveform.set_valid_flag(valid_flag)

    @staticmethod
    def _transfer_surface_type_data(l1: Level1bData, fdr_ds: xr.DataTree):

        # byte surface_type(time=54000);
        #   :_FillValue = 127B; // byte
        #   :units = 1L; // long
        #   :long_name = "surface type from GSHHG";
        #   :flag_values = "0b, 1b, 2b, 3b, 4b";
        #   :flag_meanings = "ocean land lake island_in_lake pond_in_island";
        #   :coordinates = "longitude latitude";
        #   :comment = "GSHHG data version 2.3.7";
        #   :valid_min = "0b";
        #   :valid_max = "4b";

        surface_type = fdr_ds.main.surface_type.values
        flag_map_dict = {
            "ocean": 0,
            "land": 1,
            "closed_sea": 2,
            "land_ice": -1,   # Does not exist
        }
        for key in ESA_SURFACE_TYPE_DICT.keys():
            flag = surface_type == flag_map_dict[key]
            l1.surface_type.add_flag(flag, key)

    def _transfer_range_corrections(self, l1: Level1bData, fdr_ds: xr.DataTree) -> None:
        """
        Transfer range correction data from the FDR netCDF to the l1bdata object.

        :param l1: Level-1 data object to store the transferred data
        :param fdr_ds: xarray DataTree object containing the FDR4ALT data

        return: None (l1 is changed in place)
        """
        target_dict = self.cfg.range_correction_targets
        for parameter_name in target_dict.keys():
            l1.correction.set_parameter(parameter_name, np.full(l1.time_orbit.timestamp.shape, 0.0))

    def _transfer_classifiers(self, l1: Level1bData, fdr_ds: xr.DataTree) -> None:
        """
        Transfer classifier parameter from the FDR netCDF to the l1bdata object.

        :param l1: Level-1 data object to store the transferred data
        :param fdr_ds: xarray DataTree object containing the FDR4ALT data

        return: None (l1 is changed in place)
        """
        target_dict = self.cfg.classifier_targets
        for parameter_name in target_dict.keys():
            nc_parameter_name = target_dict[parameter_name]
            nc_parameter = fdr_ds[nc_parameter_name].values
            l1.classifier.add(nc_parameter, parameter_name)


class EnvisatSGDRNC(Level1PInputHandlerBase):
    """ Converts an Envisat SGDR object into a L1bData object """

    def __init__(self,
                 cfg: Union[Dict, AttrDict],
                 raise_on_error: bool = False
                 ) -> None:
        """
        Input handler for Sentinel-3 L2WAT netCDF files from the CODA.

        :param cfg: Options from the corresponding Level-1 pre-processor config file
        :param raise_on_error: Boolean value if the class should raise an exception upon an error (default: False)
        """

        cls_name = self.__class__.__name__
        super(EnvisatSGDRNC, self).__init__(cfg, raise_on_error, cls_name)

        # Debug variables
        self.timer = None

        # Properties
        self.sgdr = None
        self.filepath = None
        self.l1 = None

    def get_l1(self,
               filepath: Union[str, Path],
               polar_ocean_check: Any = None,
               raise_on_error: bool = False
               ) -> "Level1bData":
        """
        Read the Envisat SGDR file and transfers its content to a Level1Data instance

        :param raise_on_error:
        :param polar_ocean_check:
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
        self.sgdr = ReadNC(self.filepath, nan_fill_value=True)

        # Get metadata
        self._set_input_file_metadata()

        # Polar ocean check passed, now create all l1 data groups
        self._set_l1_data_groups()

        self.timer.stop()
        logger.info("- Created L1 object in %.3f seconds" % self.timer.get_seconds())

        return self.l1

    def _set_input_file_metadata(self) -> None:
        """ Extract essential metadata information from SGDR file """
        info = self.l1.info
        sgdr = self.sgdr
        info.set_attribute("pysiral_version", PSRLCFG.version)
        info.set_attribute("mission", "envisat")
        info.set_attribute("mission_data_version", sgdr.software_version)
        info.set_attribute("orbit", sgdr.absolute_orbit_number)
        info.set_attribute("cycle", sgdr.cycle_number)
        info.set_attribute("mission_data_source", sgdr.product_name)
        info.set_attribute("timeliness", self.cfg.timeliness)

    def _set_l1_data_groups(self) -> None:
        """
        Transfer data from the necessary data groups
        :return:
        """
        self._transfer_timeorbit()            # (lon, lat, alt, time)
        self._transfer_waveform_collection()  # (power, range)
        self._transfer_range_corrections()    # (range corrections)
        self._transfer_surface_type_data()    # (land flag, ocean flag, ...)
        self._transfer_classifiers()          # (beam parameters, flags, ...)

    def _transfer_timeorbit(self) -> None:
        """
        Extracts the time/orbit data group from the SGDR data
        :return:
        """

        # Transfer the orbit position
        # and ensure longitude is -180:180 degrees east
        longitude_degrees_east = (self.sgdr.lon_20 + 180) % 360 - 180
        self.l1.time_orbit.set_position(longitude_degrees_east, self.sgdr.lat_20, self.sgdr.alt_20)

        # Transfer the timestamp
        sgdr_timestamp = self.sgdr.time_20
        units = self.cfg.sgdr_timestamp_units
        calendar = self.cfg.sgdr_timestamp_calendar
        timestamp = num2pydate(sgdr_timestamp, units, calendar)
        self.l1.time_orbit.timestamp = timestamp

        # Mandatory antenna pointing parameter (but not available for ERS)
        dummy_angle = np.full(timestamp.shape, 0.0)
        self.l1.time_orbit.set_antenna_attitude(dummy_angle, dummy_angle, dummy_angle)

        # Update meta data container
        self.l1.update_data_limit_attributes()

    def _transfer_waveform_collection(self):
        """
        Transfers the waveform data (power & range for each range bin)
        :return:
        """

        # Transfer the reformed 18Hz waveforms
        # "waveform samples (I2+Q2, 1/2048 FFT power unit): 18 Hz Ku band";
        # "the echo is corrected for the intermediate frequency filter effect";
        wfm_power = self.sgdr.waveform_fft_20_ku
        n_records, n_range_bins = wfm_power.shape

        # In Envisat data v3.0 the tracker range is already corrected for all internal effects,
        # but the nominal tracking bin is variable.
        nominal_tracking_bin = (63 + self.sgdr.offset_tracking_20/256).astype(int)
        window_delay_m = self.sgdr.tracker_range_20_ku - nominal_tracking_bin * self.cfg.bin_width_meter

        # Compute the range value for each range bin of the 18hz waveform
        wfm_range = get_envisat_wfm_range(
            window_delay_m,
            n_range_bins,
            bin_width_meter=self.cfg.bin_width_meter
        )

        # Transfer data to the waveform group
        self.l1.waveform.set_waveform_data(wfm_power, wfm_range, self.cfg.radar_mode)

        # Set valid flag to exclude calibration data
        # (see section 3.5 of Reaper handbook)
        valid_flag = np.logical_not(self.sgdr.waveform_fault_id_20.astype(bool))
        self.l1.waveform.set_valid_flag(valid_flag)

    def _transfer_range_corrections(self):
        """
        Transfer range correction data from the SGDR netCDF to the
        l1bdata object. The parameters are defined in
        config/mission_def.yaml for ers1/ers2
        -> settings.sgdr_range_correction_targets

        For a description of the parameter see section 3.10 in the
        REAPER handbook
        """

        # Get the reference times for interpolating the range corrections from 1Hz -> 20Hz
        time_1Hz = np.array(self.sgdr.time_01)
        time_20Hz = np.array(self.sgdr.time_20)

        # Loop over all range correction in config file
        grc_dict = self.cfg.range_correction_targets
        for name in grc_dict.keys():

            # Get the variable
            target_parameter = grc_dict[name]
            if target_parameter is None:
                continue
            correction = np.array(getattr(self.sgdr, target_parameter))

            # Debug code
            # -> in this case discard the variable
            n_nans = len(np.where(np.isnan(correction))[0])
            # TODO: 500 is hardcoded
            if 500 < n_nans < len(correction):
                msg = "Significant number of NaNs (%g) in range correction variable: %s"
                msg %= (n_nans, target_parameter)
                logger.warning(msg)
            elif n_nans == len(correction):
                msg = "All-NaN array encountered in range correction variable: %s"
                msg %= target_parameter
                logger.warning(msg)

            # Some Envisat range corrections are 1Hz others 20Hz
            # -> Those with "_01" in the variable name need to be
            # extrapolated to 20 Hz
            error = False
            if re.search(self.cfg.variable_identifier_1Hz, target_parameter):
                correction, error = self.interp_1Hz_to_20Hz(correction, time_1Hz, time_20Hz,
                                                            fill_on_error_value=0.0)
            if error:
                msg = f"Failing to create 20Hz range correction variable for {target_parameter}"
                logger.warning(msg)

            # Interpolate NaN's or return a zero-filled array for all-nan input variables
            correction_filtered = self.find_and_interpolate_nans(correction, fill_on_error_value=0.0)

            # Debug code
            # -> in this case discard the variable
            n_nans = len(np.where(np.isnan(correction_filtered))[0])
            if n_nans > 0:
                msg = "Remaining NaN's after filtering in %s" % target_parameter
                logger.warning(msg)

            # The range correction for the DAC follows a different convention. It is supposed to be
            # "provided as a correction to the inverted barometer correction (inv_bar_cor_01)", but in reality
            # includes the change if sea level from atmospheric pressure changes. Therefore, IB and DAC are
            # of the same magnitude in the SGDR v3.0, but inverted. To apply DAC instead of IB, the DAC must be
            # inverted.

            # Apply inversion to both `hf_fluct_cor_01` and `hf_fluct_cor_reanalysis_01`
            if re.search("hf_fluct", target_parameter):
                correction_filtered *= -1.0

            # Set the parameter
            self.l1.correction.set_parameter(name, correction_filtered)

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
            nc_parameter = getattr(self.sgdr, nc_parameter_name)
            self.l1.classifier.add(nc_parameter.flatten(), parameter_name)

    def _transfer_surface_type_data(self):
        surface_type = self.sgdr.surf_class_20
        for key in ESA_SURFACE_TYPE_DICT.keys():
            flag = surface_type == ESA_SURFACE_TYPE_DICT[key]
            self.l1.surface_type.add_flag(flag, key)

    @staticmethod
    def find_and_interpolate_nans(variable, fill_on_error_value=np.nan):
        """
        Replace NaN's in variable with linear interpolated values
        :param variable:
        :return: interpolated variable
        """
        is_nan = np.isnan(variable)
        n_nans = len(np.where(is_nan)[0])
        if n_nans == 0:
            variable_filtered = variable
        else:
            x = np.arange(len(variable))
            valid = np.where(np.logical_not(is_nan))[0]
            try:
                f = interpolate.interp1d(x[valid], variable[valid], fill_value="extrapolate",
                                         bounds_error=False)
                variable_filtered = f(x)
            except ValueError:
                variable_filtered = np.full(variable.shape, fill_on_error_value)
        return variable_filtered

    @staticmethod
    def interp_1Hz_to_20Hz(variable_1Hz, time_1Hz, time_20Hz, fill_on_error_value=np.nan, **kwargs):
        """
        Computes a simple linear interpolation to transform a 1Hz into a 20Hz variable
        :param variable_1Hz: an 1Hz variable array
        :param time_1Hz: 1Hz reference time
        :param time_20Hz: 20 Hz reference time
        :return: the interpolated 20Hz variable
        """
        error_status = False
        try:
            is_valid = np.logical_and(np.isfinite(time_1Hz), np.isfinite(variable_1Hz))
            valid_indices = np.where(is_valid)[0]
            f = interpolate.interp1d(time_1Hz[valid_indices], variable_1Hz[valid_indices],
                                     fill_value="extrapolate", bounds_error=False, **kwargs)
            variable_20Hz = f(time_20Hz)
        except ValueError:
            variable_20Hz = np.full(time_20Hz.shape, fill_on_error_value)
            error_status = True
        return variable_20Hz, error_status

    @property
    def empty(self):
        """
        Default return object, if nodata should be returned
        :return: Representation of an empty object (None)
        """
        return None
