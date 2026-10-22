# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


from pathlib import Path
from typing import Dict, Optional
from cftime import num2pydate
from pydantic import BaseModel

import numpy as np
import pandas as pd
import xarray as xr

from pysiral import PSRLCFG
from pysiral.core.flags import ESA_SURFACE_TYPE_DICT
from pysiral.mission import SourceDataLoader
from pysiral.mission.envisat.functions import get_envisat_wfm_range
from pysiral.l1.l1_data import Level1bData


class FDR4ALTConfig(BaseModel):
    """
    Configuration for Envisat FDR4ALT data loader.
    """
    name: str
    mission: str
    classifier_targets: Dict[str, str]
    range_correction_targets: Dict[str, str]
    range_bin_width_envisat: float = 0.4686
    range_bin_width_ers: float = 0.4545
    nominal_tracking_bin_ers: float = 32.5
    radar_mode: str = "lrm"
    timeliness: str = "rep"
    raise_on_error: bool = True


class FDR4ALTDataLoader(object):

    def __init__(self, **kwargs) -> None:
        self.cfg = FDR4ALTConfig(**kwargs)

    def _set_input_file_metadata(self, l1: Level1bData, fdr_ds: xr.DataTree, mission_id: str) -> None:
        """
        Extract dataset attributes from FDR dataset and store them
        in the Level-1 data object.

        :param l1: Level-1 data object to store the metadata
        :param fdr_ds: xarray DataTree object containing the FDR4ALT data

        :return: None (l1 is change in place)
        """

        mission = {"er1": "ers1", "er2": "ers2", "en1": "envisat"}.get(mission_id.lower(), mission_id)
        l1.info.set_attribute("pysiral_version", PSRLCFG.version)
        l1.info.set_attribute("mission", mission)
        l1.info.set_attribute("mission_data_version", fdr_ds.attrs.get("source"))
        l1.info.set_attribute("orbit", fdr_ds.attrs.get("pass_number"))
        l1.info.set_attribute("cycle", fdr_ds.attrs.get("cycle_number"))
        l1.info.set_attribute("mission_data_source", fdr_ds.attrs.get("title"))
        l1.info.set_attribute("timeliness", self.cfg.timeliness)

    def _get_l1(self, source_file: pd.Series, **kwargs) -> Optional[Level1bData]:
        """
        Read the Envisat FDR4ALT file and transfers its content to a Level1Data instance

        :param source_file: A row from the input file catalog DataFrame representing a source file.
        :return: The parsed (or empty) Level-1 data container
        """

        l1 = Level1bData()
        # NOTE: `decode_times=False` prevents a pandas.errors.OutOfBoundsTimedelta exception
        with xr.open_datatree(source_file.filepath, decode_times=False) as fdr_ds:
            self._set_input_file_metadata(l1, fdr_ds, mission_id=source_file.mission_id)
            self._transfer_data_groups(l1, fdr_ds)
        return l1

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
        l1.classifier.add(fdr_ds.main.time, "time_num")

        # Update meta data container
        l1.update_data_limit_attributes()

    def _transfer_waveform_collection(self, l1: Level1bData, fdr_ds: xr.DataTree) -> None:
        # To be overridden by subclasses
        pass

    def _transfer_range_corrections(self, l1: Level1bData, fdr_ds: xr.DataTree) -> None:
        """
        Transfer range correction data from the FDR netCDF to the l1bdata object.

        :param l1: Level-1 data object to store the transferred data
        :param fdr_ds: xarray DataTree object containing the FDR4ALT data

        :return: None (l1 is changed in place)
        """
        target_dict = self.cfg.range_correction_targets
        for parameter_name in target_dict.keys():
            l1.correction.set_parameter(parameter_name, np.full(l1.time_orbit.timestamp.shape, np.nan))

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


class EnvisatFDR4ALTDataLoader(SourceDataLoader, FDR4ALTDataLoader):

    def __init__(self, **kwargs) -> None:
        """
        Input handler for Envisat FDR4ALT netCDF files combining the altimeter data from
        Fundamental Data Record (FDR) and the range correction (tides & atmospheric corrections)
        from the thematic sea ice product.

        :param cfg: Options from the corresponding Level-1 pre-processor config file
        :param raise_on_error: Boolean value if the class should raise an exception upon an error (default: False)
        """
        # Properties
        kwargs.update({"mission": "envisat"})
        super().__init__(**kwargs)

    def get_l1(self, filepath: Path, **kwargs) -> Optional[Level1bData]:
        """
        Mandatory method for Dataloader ABC, but implemented in general FDR4ALTDataLoader class.

        :param filepath:
        :param kwargs:

        :return: Optional[Level1bData]
        """
        return super()._get_l1(filepath, **kwargs)

    def _transfer_waveform_collection(self, l1: Level1bData, fdr_ds: xr.DataTree) -> None:

        # Transfer the reformed 18Hz waveforms
        # "waveform samples (I2+Q2, 1/2048 FFT power unit): 18 Hz Ku band";
        # "the echo is corrected for the intermediate frequency filter effect";
        wfm_power = fdr_ds.main.Ku.power_waveform.values
        n_records, n_range_bins = wfm_power.shape

        # From the FDR4ALT product handbook
        # 6.2.8 Offset tracking flag and reference Abscissa (ENVISAT)
        chirp_band_width_dict = {
            0: 320_000_000.,   # Ku band Ocean Mode
            1:  82_500_000,    # Lu band Ice Mode
            2:  20_000_000,    # ?
        }

        # NOTE: There may be NaN's values in chirp band, which will be replaced by the default value of 320 MHz
        #       lacking a better solution at this point (and assuming it won't have practical consequences)
        bandwidth = np.vectorize(chirp_band_width_dict.get)(fdr_ds.main.Ku.chirp_band.values, 320_000_000.0)
        nominal_tracking_bin = 63 + fdr_ds.main.offset_tracking.values/256 * bandwidth / 320_000_000.0

        # Window delay is defined as the range to the start of the range window
        window_delay_m = fdr_ds.main.Ku.tracker_range.values - nominal_tracking_bin * self.cfg.range_bin_width_envisat

        # Compute the range value for each range bin of the 18hz waveform
        wfm_range = get_envisat_wfm_range(
            window_delay_m,
            n_range_bins,
            bin_width_meter=self.cfg.range_bin_width_envisat
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


class ERSFDR4ALTDataLoader(SourceDataLoader, FDR4ALTDataLoader):

    def __init__(self, **kwargs) -> None:
        """
        Input handler for ERS-1/2 FDR4ALT netCDF files combining the altimeter data from
        Fundamental Data Record (FDR) and the range correction (tides & atmospheric corrections)
        from the thematic sea ice product.

        :param cfg: Options from the corresponding Level-1 pre-processor config file
        :param raise_on_error: Boolean value if the class should raise an exception upon an error (default: False)
        """
        # Properties
        kwargs.update({"mission": "ers"})
        super().__init__(**kwargs)

    def get_l1(self, filepath: Path, **kwargs) -> Optional[Level1bData]:
        """
        Mandatory method for Dataloader ABC, but implemented in general FDR4ALTDataLoader class.

        :param filepath:
        :param kwargs:
        :return: Optional[Level1bData]
        """
        return super()._get_l1(filepath, **kwargs)

    def _transfer_waveform_collection(self, l1: Level1bData, fdr_ds: xr.DataTree) -> None:

        # Transfer the reformed 18Hz waveforms
        # "waveform samples (I2+Q2, 1/2048 FFT power unit): 18 Hz Ku band";
        # "the echo is corrected for the intermediate frequency filter effect";
        wfm_power = fdr_ds.main.power_waveform.values
        n_records, n_range_bins = wfm_power.shape

        # # Window delay is defined as the range to the start of the range window
        window_delay_m = fdr_ds.main.tracker_range.values - self.cfg.nominal_tracking_bin_ers * self.cfg.range_bin_width_ers
        #
        # Compute the range value for each range bin of the 18hz waveform
        wfm_range = get_envisat_wfm_range(
            window_delay_m,
            n_range_bins,
            bin_width_meter=self.cfg.range_bin_width_ers
        )
        # Transfer data to the waveform group
        l1.waveform.set_waveform_data(wfm_power, wfm_range, self.cfg.radar_mode)

        # Set valid flag to exclude calibration data
        all_flags = np.stack(
            (
                fdr_ds.main.qual_wf_not_tracking.astype(bool),
                fdr_ds.main.qual_wf_noisy.astype(bool),
                fdr_ds.main.qual_wf_low_var.astype(bool),
                fdr_ds.main.qual_wf_bad_le.values.astype(bool),
                fdr_ds.main.qual_range_cor_net_instr.values.astype(bool),
            )
        )
        valid_flag = np.logical_not(np.max(all_flags, axis=0))
        l1.waveform.set_valid_flag(valid_flag)
