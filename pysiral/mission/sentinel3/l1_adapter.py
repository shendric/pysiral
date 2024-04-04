
import contextlib
from pathlib import Path

import numpy as np
import xarray
import xmltodict
from cftime import num2pydate
from loguru import logger
from scipy import interpolate
from typing import Optional

from pysiral import __version__ as pysiral_version
from pysiral.core.clocks import StopWatch
from pysiral.core.flags import ESA_SURFACE_TYPE_DICT
from pysiral.core.helper import parse_datetime_str
from pysiral.l1data import Level1bData
from pysiral.l1 import SourceDataLoader


class Sentinel3L2SeaIce(
    SourceDataLoader,
    supported_source_datasets=[
        "sentinel3a_rep_l2lansi_v5p0",
        "sentinel3a_nrt_l2lansi_v5p0",
        "sentinel3b_rep_l2lansi_v5p0",
        "sentinel3b_nrt_l2lansi_v5p0"
    ]
):

    def __init__(self, cfg, raise_on_error=False):
        """
        Input handler for Sentinel-3 L2WAT netCDF files from the CODA.
        :param cfg: A treedict object (root.input_handler.options) from the corresponding Level-1 pre-processor
                    config file
        :param raise_on_error: Boolean value if the class should raise an exception upon an error (default: False)
        """

        # Init main class variables
        self.cfg = cfg
        self.raise_on_error = raise_on_error
        self.nc = None
        self.filepath = None
        self.l1 = None

        # Debug variables
        self.timer = None

    def get_l1(self, filepath, polar_ocean_check=None) -> Optional[Level1bData]:
        """
        Create a Level-1 data container from Sentinel-3 CODA L2WAT files
        :param filepath: The full file path to the netCDF file
        :param polar_ocean_check:
        :return: The parsed (or empty) Level-1 data container
        """

        #  for debug purposes
        self.timer = StopWatch()
        self.timer.start()

        # Save filepath
        self.filepath = filepath
        logger.info(f"- {filepath.parts[-2]}")

        # Create an empty Level-1 data object
        self.l1 = Level1bData()

        # Input Validation
        if not Path(filepath).is_file():
            raise ValueError(f"Not a valid file: {filepath}")

        # Parse xml header file
        with contextlib.suppress(FileNotFoundError):
            self._parse_xml_manifest(filepath)

        # Parse the input netCDF file
        self._read_input_netcdf(filepath)
        if self.nc is None:
            return self.empty

        # Get metadata
        self._set_input_file_metadata()

        # Test if input file contains data over polar oceans (optional)
        if polar_ocean_check is not None:
            has_polar_ocean_data = polar_ocean_check.has_polar_ocean_segments(self.l1.info)
            if not has_polar_ocean_data:
                self.timer.stop()
                return self.empty

        # Polar ocean check passed, now fill the remaining l1 data groups
        self._set_l1_data_groups()

        self.timer.stop()
        logger.info("- Created L1 object in %.3f seconds" % self.timer.get_seconds())

        # Return the l1 object
        return self.l1

    @staticmethod
    def interp_01_hz_to_20_hz(variable_01_hz, time_01_hz, time_20_hz, **kwargs):
        """
        Computes a simple linear interpolation to transform a 1Hz into a 20Hz variable
        :param variable_01_hz: an 1Hz variable array
        :param time_01_hz: 1Hz reference time
        :param time_20_hz: 20 Hz reference time
        :return: the interpolated 20Hz variable
        """
        error_status = False
        try:
            f = interpolate.interp1d(time_01_hz, variable_01_hz, bounds_error=False, **kwargs)
            variable_20_hz = f(time_20_hz)
        except ValueError:
            fill_value = np.nan
            variable_20_hz = np.full(time_20_hz.shape, fill_value)
            error_status = True
        return variable_20_hz, error_status

    @staticmethod
    def parse_sentinel3_l1b_xml_header(filename):
        """
        Reads the XML header file of a Sentinel 3 L1b Data set
        and returns the contents as an OrderedDict
        """
        with open(str(filename)) as fd:
            content_odereddict = xmltodict.parse(fd.read())
        return content_odereddict[u'xfdu:XFDU']

    def _parse_xml_manifest(self, filepath):
        """
        Parse the Sentinel-3 XML header file and extract key attributes for filtering
        :param filepath: the filepath for the netcdf
        :return: None
        """
        # Retrieve header information from mission settings
        xml_header_file = self.cfg.xml_manifest
        filename_header = Path(filepath).parent / xml_header_file
        self._xmlh = self.parse_sentinel3_l1b_xml_header(filename_header)

    def _get_xml_content(self, section_name, tag):
        """ Returns the generalProductInformation content of the xml manifest
        :return: dictionary
        """

        # Extract Metadata
        metadata = self._xmlh["metadataSection"]["metadataObject"]

        # Extract General Product Info
        index = self.cfg.xml_metadata_object_index[section_name]
        product_info = metadata[index]["metadataWrap"]["xmlData"]
        return product_info[tag]

    def _read_input_netcdf(self, filepath):
        """
        Read the netCDF file via xarray
        :param filepath: The full filepath to the netCDF file
        :return: none
        """
        try:
            self.nc = xarray.open_dataset(filepath, decode_times=False, mask_and_scale=True)
        except:
            logger.error(f"Error encountered by xarray parsing: {filepath}")
            self.nc = None

    def _set_input_file_metadata(self):
        """
        Populates the product info segment of the Level1Data object with information from
        the global attributes of the netCDF and content of the xml manifest
        :return: None
        """

        # Shortcuts
        metadata = self.nc.attrs
        info = self.l1.info

        # Get xml manifest content
        product_info = self._get_xml_content(
            "generalProductInformation",
            "sentinel3:generalProductInformation")
        sral_info = self._get_xml_content(
            "sralProductInformation",
            "sral:sralProductInformation"
        )

        # Processing environment metadata
        info.set_attribute("pysiral_version", pysiral_version)

        # General product metadata
        mission = metadata["mission_name"].lower().replace(" ", "")
        info.set_attribute("mission", str(mission))
        info.set_attribute("mission_sensor", "sral")
        info.set_attribute("mission_data_version", metadata["source"])
        info.set_attribute("orbit", metadata["absolute_rev_number"])
        info.set_attribute("cycle", metadata["cycle_number"])
        info.set_attribute("mission_data_source",  metadata["product_name"])
        info.set_attribute("timeliness", self.cfg.timeliness_dict[str(product_info["sentinel3:timeliness"])])

        # Time-Orbit Metadata
        lats = [float(metadata["first_meas_lat"]), float(metadata["last_meas_lat"])]
        lons = [float(metadata["first_meas_lon"]), float(metadata["last_meas_lon"])]
        start_time = parse_datetime_str(metadata["first_meas_time"])
        stop_time = parse_datetime_str(metadata["last_meas_time"])
        info.set_attribute("start_time", start_time.replace(tzinfo=None))
        info.set_attribute("stop_time", stop_time.replace(tzinfo=None))
        info.set_attribute("lat_min", np.amin(lats))
        info.set_attribute("lat_max", np.amax(lats))
        info.set_attribute("lon_min", np.amin(lons))
        info.set_attribute("lon_max", np.amax(lons))

        # Product Content Metadata
        for mode in ["sar", "sin", "lrm"]:
            percent_value = 0.0
            if mode == "sar":
                percent_value = 100.
            info.set_attribute(f"{mode}_mode_percent", percent_value)
        info.set_attribute("open_ocean_percent", float(sral_info["sral:openOceanPercentage"]))

    def _set_l1_data_groups(self):
        """
        Fill all data groups of the Level-1 data object with the content of the netCDF file. This is just the
        overview method, see specific sub-methods below
        :return: None
        """
        self._set_time_orbit_data_group()
        self._set_waveform_data_group()
        self._set_range_correction_group()
        self._set_surface_type_group()
        self._set_classifier_group()

    def _set_time_orbit_data_group(self):
        """
        Transfer the time orbit parameter from the netcdf to l1 data object
        :return: None
        """

        # Transfer the timestamp
        # NOTE: Here it is critical that the xarray does not automatically decodes time since it is
        #       difficult to work with the numpy datetime64 date format. Better to compute datetimes using
        #       a know num2pydate conversion
        utc_timestamp = num2pydate(self.nc.time_20_ku.values, units=self.nc.time_20_ku.units)
        self.l1.time_orbit.timestamp = utc_timestamp

        # Set the geolocation
        self.l1.time_orbit.set_position(
            self.nc.lon_20_ku.values,
            self.nc.lat_20_ku.values,
            self.nc.alt_20_ku.values,
            self.nc.orb_alt_rate_20_ku.values)

        # Set antenna attitude
        # NOTE: These are only available in 1Hz and need to be interpolated
        time_01, time_20 = self.nc.time_01.values, self.nc.time_20_ku.values
        pitch_angle_20, stat = self.interp_01_hz_to_20_hz(
            self.nc.off_nadir_pitch_angle_pf_01.values,
            time_01,
            time_20)
        roll_angle_20, stat = self.interp_01_hz_to_20_hz(
            self.nc.off_nadir_roll_angle_pf_01.values,
            time_01,
            time_20)
        yaw_angle_20, stat = self.interp_01_hz_to_20_hz(
            self.nc.off_nadir_yaw_angle_pf_01.values,
            time_01,
            time_20)
        self.l1.time_orbit.set_antenna_attitude(pitch_angle_20, roll_angle_20, yaw_angle_20)

    def _set_waveform_data_group(self):
        """
        Transfer of the waveform group to the Level-1 object. This includes
          1. the computation of waveform power in Watts
          2. the computation of the window delay in meter for each waveform bin
          3. extraction of the waveform valid flag
        :return: None
        """

        # Get the waveform
        # NOTE: The waveform is given in counts
        wfm_counts = self.nc.waveform_20_ku.values
        n_records, n_range_bins = wfm_counts.shape

        # -- Waveform to power conversion is not possible --
        # FAQ-Altimetry-003 - The waveform contained in the enhanced data file is scaled waveform.
        # How can the scaled waveform be converted to power waveform?
        # It is not possible to convert power waveform in watts because it requires some instrumental
        # information from the industry who designed the altimeter and this information is not available
        # to the users. Nevertheless, the S3A waveforms can be compared to each other, by applying the
        # agc values provided in the SRAL L1 products.
        # (https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-3-altimetry/appendices/faq)

        # NOTE: Make sure type of waveform is double (hard requirement for cythonized retrackers)
        wfm_power = wfm_counts.astype(np.float64)

        # Get the window delay
        # "The tracker_range_20hz is the range measured by the onboard tracker
        #  as the window delay, corrected for instrumental effects and
        #  CoG offset"
        tracker_range_20hz = self.nc.tracker_range_20_ku.values
        wfm_range = np.ndarray(shape=wfm_counts.shape, dtype=np.float64)
        range_bin_index = np.arange(n_range_bins)
        for record in np.arange(n_records):
            wfm_range[record, :] = tracker_range_20hz[record] + \
                (range_bin_index * self.cfg.range_bin_width) - \
                (self.cfg.nominal_tracking_bin * self.cfg.range_bin_width)

        # Set the operation mode
        op_mode = self.nc.instr_op_mode_20_ku.values
        op_mode_translator = self.cfg.instr_op_mode_list
        radar_mode = np.array([op_mode_translator[int(val)] for val in op_mode]).astype("int8")

        # Set the waveform
        self.l1.waveform.set_waveform_data(wfm_power, wfm_range, radar_mode)

        # Get the valid flags
        # TODO: Find a way to get a valid flag
        # measurement_confident_flag = self.nc.flag_mcd_20_ku.values
        # valid_flag = measurement_confident_flag == 0
        # self.l1.waveform.set_valid_flag(valid_flag)

    def _set_range_correction_group(self):
        """
        Transfer the range corrections defined in the l1p config file to the Level-1 object
        NOTE: The range corrections are all in 1 Hz and must be interpolated to 20Hz
        :return: None
        """

        # Get the reference times for interpolating the range corrections from 1Hz -> 20Hz
        time_1_hz = self.nc.time_01.values
        time_20_hz = self.nc.time_20_ku.values

        # Loop over all range correction variables defined in the processor definition file
        keys = self.cfg.range_correction_targets.keys()
        for key in keys:
            var_name = self.cfg.range_correction_targets[key]
            variable = getattr(self.nc, var_name)
            if variable.values.size == time_1_hz.size:
                variable_20_hz, error_status = self.interp_01_hz_to_20_hz(variable.values, time_1_hz, time_20_hz)
            else:
                error_status = False
                variable_20_hz = variable.values
            if error_status:
                msg = f"- Error in 20Hz interpolation for variable `{var_name}` -> set only dummy"
                logger.warning(msg)
            self.l1.correction.set_parameter(key, variable_20_hz)

        # Sentinel-3 specific item: The dynamic atmosphere correction (`hf_fluct_cor_01`)
        # is provided as 'Provided as a correction to the inverted barometer correction (inv_bar_cor_01)'
        # Therefore, the dynamic_atmosphere field needs to update.
        if "dynamic_atmosphere" in keys and "inverse_barometric" in keys:
            logger.debug("- Update dac")
            true_dac = self.l1.correction.dynamic_atmosphere + self.l1.correction.inverse_barometric
            self.l1.correction.set_parameter("dynamic_atmosphere", true_dac)

    def _set_surface_type_group(self):
        """
        Transfer of the surface type flag to the Level-1 object

        :return: None
        """

        # Set the flag
        for key in ESA_SURFACE_TYPE_DICT.keys():
            flag = self.nc.surf_type_20_ku.values == ESA_SURFACE_TYPE_DICT[key]
            self.l1.surface_type.add_flag(flag, key)

    def _set_classifier_group(self):
        """
        Transfer the classifiers defined in the l1p config file to the Level-1 object.
        NOTE: It is assumed that all classifiers are 20Hz
        In addition, a few legacy parameter are computed based on the waveform counts that is only available at
        this stage. Computation of other parameter such as sigma_0, leading_edge_width, ... are moved to the
        post-processing
        :return: None
        """
        # Loop over all classifier variables defined in the processor definition file
        time_01, time_20 = self.nc.time_01.values, self.nc.time_20_ku.values
        for key, target in self.cfg.classifier_targets.items():
            if "01" in target:
                variable_01_hz = getattr(self.nc, target)
                variable_20_hz, _ = self.interp_01_hz_to_20_hz(variable_01_hz, time_01, time_20)
            else:
                variable_20_hz = getattr(self.nc, target)
            self.l1.classifier.add(variable_20_hz, key)

    @property
    def empty(self):
        """
        Default return object, if nodata should be returned
        :return: Representation of an empty object (None)
        """
        return None
