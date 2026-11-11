# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 13:57:56 2016

@author: Stefan

Important Note:

    All sitype data handlers must be subclasses of pysiral.auxdata.AuxdataBaseClass in order to work
    for the Level-2 Processor. If the auxiliary class is based on a static dataset, this should be parsed
    in `__init__`.

    Please review the variables and properties in the parent class, as well as the correspodning config and
    support classes for grid track interpolation in the pysiral.auxdata module for additional guidance.

    The only other hard requirements is the presence of on specific method in order to be a valid subclass of
    AuxdataBaseClass:


        get_l2_track_vars(l2)

            This method will be called during the Level-2 processor. The argument is the Level-2 data object and
            the purpose of the method is to compute the auxilary variable(s) and associated uncertainty. These
            variable need to be registered using the `register_auxvar(id, name, value, uncertainty)` method of
            the base class. All sitype subclasses need to register at minimum the following variable:

                sea ice type (fraction of multi year ice):
                    id: sitype
                    name: sea_ice_type

            e.g., this code line is mandatory for `get_l2_track_vars` (uncertainty can be None):

                # Register Variables
                self.register_auxvar("sitype", "sea_ice_type", value, uncertainty)

"""


from pathlib import Path
from loguru import logger
from typing import List

import numpy as np
from scipy import interpolate

from pysiral.auxdata import GridTrackInterpol, AuxDataLoader, AuxDataExitFlag
from pysiral.core.iotools import ReadNC
from pysiral.l2.alg.sea_level_anomaly import SLABaseFunctionality


class OsiSafSIType(AuxDataLoader):
    """ This is a class for the OSI-403 product with variables ice_type and confidence_level """

    def __init__(self, cfg, **kwargs):
        super(OsiSafSIType, self).__init__()
        self.cfg = cfg
        self._data = None
        self.start_time = None
        self.hemisphere_code = None

    def apply(self, l1, l2) -> AuxDataExitFlag:
        """ Default grid auxiliary data set"""

        exit_status = AuxDataExitFlag.NOMINAL

        # These properties are needed to construct the product path
        self.start_time = l2.info.start_time
        self.hemisphere_code = l2.hemisphere_code

        # Set the requested data
        self.set_requested_date_from_l2(l2)

        # Update the external data
        self.update_external_data()

        # Check if data is available
        if self._data is None:
            sitype = self.get_empty_array(l2)
            uncertainty = self.get_empty_array(l2)
            exception_on_error = self.cfg.options.get("exception_on_error", True)
            exit_status = AuxDataExitFlag.ERROR_SKIP
            if exception_on_error:
                raise FileNotFoundError(f"Auxdata: {self.__class__.__name__}: File not found: {self.requested_filepath}")
        else:
            # Get and return the track
            sitype, uncertainty = self._get_sitype_track(l2)

        # (Optional) Fill gaps in the sea ice type product
        # where the sea-ice concentration data indicates the
        # presence of sea ice. This usually happens close to
        # coastlines since the sea-ice type product has a larger
        # land exclusion zone than SIC (which leads to loss of
        # data, e.g. in the Canadian Archipelago)
        fill_gaps = self.cfg.options.get("fill_valid_sic_gaps", False)
        if fill_gaps:
            sitype, uncertainty = fill_sitype_gaps(sitype, uncertainty, l2.sic[:])

        # Register the data
        l2.set_auxiliary_parameter("sitype", "sea_ice_type", sitype, uncertainty)

        return exit_status

    def load_requested_auxdata(self) -> None:
        """
        Required subclass method: Load the data file necessary to satisfy condition for requested date
        """
        path = Path(self.requested_filepath)
        if not path.is_file():
            logger.error(f"OsiSafSIType: File not found: {path}")
            self._data = None
            return
        self._data = ReadNC(path)
        logger.info(f"OsiSafSIType: Loaded SIType file: {path}")

    def _get_sitype_track(self, l2):

        # Extract from grid
        griddef = self.cfg.options[l2.hemisphere]
        grid_lons, grid_lats = self._data.lon, self._data.lat
        grid2track = GridTrackInterpol(l2.track.longitude, l2.track.latitude, grid_lons, grid_lats, griddef)
        sitype = grid2track.get_from_grid_variable(self._data.ice_type[0, :, :], flipud=True)

        # set fill values to flag 0 -> nan
        fillvalues = np.where(sitype == -1)[0]
        sitype[fillvalues] = 0

        # --- Translate sea-ice type codes into myi fraction ---
        # flag_meanings: -1: fill value, 1: open_water, 2: first_year_ice, 3: multi_year_ice, 4: ambiguous
        translator = np.array([np.nan, np.nan, 0.0, 1.0, 0.5])
        sitype = np.array([translator[value] for value in sitype])

        # --- Get Uncertainty ---

        # NOTE: There has been a change in OSI-403-d, when the `confidence_level`
        #       variable was replaced by uncertainty.
        try:
            # Translate confidence level into myi fraction uncertainty
            # flag_meaning: 0: unprocessed, 1: erroneous, 2: unreliable, 3: acceptable, 4: good, 5: excellent
            confidence_level = grid2track.get_from_grid_variable(self._data.confidence_level[0, :, :], flipud=True)
            translator = np.array([np.nan, 1., 0.5, 0.2, 0.1, 0.0])
            sitype_uncertainty = np.array([translator[value] for value in confidence_level])
        except AttributeError:
            sitype_uncertainty = grid2track.get_from_grid_variable(self._data.uncertainty[0, :, :], flipud=True)

        return sitype, sitype_uncertainty

    @property
    def requested_filepath(self):
        """ Note: this overwrites the property in the super class due to some
        peculiarities with the filenaming (hemisphere code) """
        path = Path(self.cfg.local_repository)
        for subfolder_tag in self.cfg.subfolders:
            subfolder = getattr(self, subfolder_tag)
            path = path / subfolder
        filename = self.cfg.filenaming.format(
            year=self.year, month=self.month, day=self.day,
            hemisphere_code=self.hemisphere_code)
        path = path / filename
        return path


class OsiSafSITypeCDR(AuxDataLoader):
    """
    Class for reprocessed OSISAF sea ice type products (e.g. for C3S). Currently supports the
    - C3S sea ice type climate data record v1.0
    - C2s sea ice type interim climate data record v1.0
    - C3S sea ice tyoe climate data record v2.0
    """

    def __init__(self, cfg) -> None:
        """
        Init the class.
        NOTE: The options template can be different for continuous data sets (is_cdr_icdr: False)
              and those with a dedicated split into a climate data record (cdr) and an interim
              climate data record (icdr). A pre-processing of the options dictionary is therefore necessary
              to follow the mechanics of the auxiliary data class.
        :param cfg: Configuration object
        """
        super(OsiSafSITypeCDR, self).__init__()

        # Pre-process the options for cdr/icdr data sets
        is_cdr_icdr = cfg.options.get("is_cdr_icdr", False)
        target_version = cfg.options.get("version", None)

        match target_version:
            case str():
                version = {'cdr': target_version, 'icdr': target_version}
            case dict():
                version = target_version
            case None:
                version = {'cdr': None, 'icdr': None}
            case _:
                raise ValueError(f"Invalid target_version: {target_version}")

        if is_cdr_icdr:
            global_options = cfg.options.get("global", {})
            version_options = cfg.options.get(version["cdr"], {})
            cfg.options.update(global_options)
            cfg.options.update(version_options)
            cfg.options.update({"long_name_template": cfg.long_name})

        self.cfg = cfg
        self._data = None
        self.start_time = None
        self.hemisphere_code = None

    def apply(self, l1, l2) -> AuxDataExitFlag:
        """
        Mandadory method of AuxdataBaseClass subclass.
        Registers two variables to the Level-2 data container:
            - MYI fraction (id: sitype, name: sea_ice_type)
            - MYI fraction uncertainty (id: sitype.uncertainty, name: sea_ice_type_uncertainty)
        :param l2: Level-2 Data object
        :return: None
        """
        exit_status = AuxDataExitFlag.NOMINAL

        # These properties are needed to construct the product path
        self.start_time = l2.info.start_time
        self.hemisphere_code = l2.hemisphere_code

        # Set the requested data
        self.set_requested_date_from_l2(l2)

        # Update the external data
        self.update_external_data()

        # Check if error with file I/O
        if self._data is None:
            sitype = self.get_empty_array(l2)
            uncertainty = self.get_empty_array(l2)
            exit_status = AuxDataExitFlag.ERROR_CONTINUE
        else:
            # Get and return the track
            sitype, uncertainty = self._get_sitype_track(l2)

        # FIXME: Is this relevant?
        # (Optional) Fill gaps in the sea ice type product
        # where the sea-ice concentration data indicates the
        # presence of sea ice. This usually happens close to
        # coastlines since the sea-ice type product has a larger
        # land exclusion zone than SIC (which leads to loss of
        # data, e.g. in the Canadian Archipelago)
        fill_gaps = self.cfg.options.get("fill_valid_sic_gaps", False)
        if fill_gaps:
            sitype, uncertainty = fill_sitype_gaps(sitype, uncertainty, l2.sic[:])

        # Register the sea ice type data to the L2 data object
        l2.set_auxiliary_parameter("sitype", "sea_ice_type", sitype, uncertainty)
        return exit_status

    def load_requested_auxdata(self) -> None:
        """
        Loads file from local repository only if needed
        :return:
        """

        # Retrieve the file path for the requested date from a property of the auxdata parent class
        path = Path(self.requested_filepath)

        # Validation
        if not path.is_file():
            logger.error(f"{self.__class__.__name__}: File not found: {path}")
            self._data = None
            return

        # Read and prepare input data
        self._data = ReadNC(path)

    def _get_sitype_track(self, l2):
        """
        Extract ice type and ice type uncertainty along the track
        :param l2:
        :return: sitype (array), sitype_uncertainty (array)
        """

        # --- Extract sea ice type and its uncertainty along the trajectory ---
        hemisphere = {"nh": "north", "sh": "south"}.get(l2.hemisphere_code, l2.hemisphere_code)
        griddef = self.cfg.options[hemisphere]
        grid_lons, grid_lats = self._data.lon, self._data.lat
        grid2track = GridTrackInterpol(l2.track.longitude, l2.track.latitude, grid_lons, grid_lats, griddef)
        sitype = grid2track.get_from_grid_variable(self._data.ice_type[0, :, :], flipud=True)
        uncertainty = grid2track.get_from_grid_variable(self._data.uncertainty[0, :, :], flipud=True)

        # --- Translate the flags in the file into MYI-fractions ---
        # The fill value of the sea ice type as changed for different product version
        # To assure backwards compatibility the fill value defaults to -1 (sea ice type v1p0)
        # an for newer versions it must be specified in the options
        fill_value = self.cfg.options.get("fill_value", -1)
        fillvalue_locations = np.where(sitype == fill_value)[0]
        sitype[fillvalue_locations] = 5
        sitype = np.array([*map(self.flag_translator, sitype)])

        # --- Ensure uncertainty units are fractions and not percent ---
        # In the sea-ice type cdr/icdr v1.0 the unit of uncertainty in percent
        # while from v2.0 on the unit is fraction. Therefore a switch has been
        # introduced for v2.0 that turn the unit conversion (default) off
        uncertainty_unit_is_percent = self.cfg.options.get("uncertainty_unit_is_percent", True)
        if uncertainty_unit_is_percent:
            uncertainty = uncertainty / 100.

        # All done, return values
        return sitype, uncertainty

    @staticmethod
    def flag_translator(flag):
        """
        Converts the flag in the file into multi-year ice (MYI) fraction:

            0.0: fyi
            0.5: ambiguous
            1.0: myi
            NaN: everything else

        This method only converts a single flag value to MYI fraction and is intended to use with map()

            myi_fraction = map(self.flag_translater, flag_values)

        Documentation of the flag from the product files (v2.0). Fill value may differ between
        sea-ice type CDR versions:

            ```
            byte ice_type(time=1, yc=432, xc=432);
                  :_FillValue = -127B; // byte
                  :long_name = "Classification of sea ice into the classes of first-year ice and multiyear ice";
                  :standard_name = "sea_ice_classification";
                  :valid_min = 1B; // byte
                  :valid_max = 4B; // byte
                  :grid_mapping = "Lambert_Azimuthal_Grid";
                  :coordinates = "time lat lon";
                  :flag_values = 1B, 2B, 3B, 4B; // byte
                  :flag_meanings = "open_water first_year_ice multi_year_ice ambiguous";
                  :flag_descriptions = "flag 1: No ice or very open ice (less than 30% ice concentration)
                                        flag 2: Seasonal ice that has formed since last melting season
                                        flag 3: Older ice that has survived at least one melting season
                                        flag 4: Ambiguous ice with non-significant classification";
                  :ancillary_variables = "uncertainty status_flag";
                  :comment = "this field is the primary sea ice type estimate for this climate data record";
            ```

        :param flag: (int) The sea ice classification flag value
        :return: myi_fraction: (float) MYI fraction
        """

        # List to translate sea ice classification flag into MYI fractions
        translator_list = np.array([np.nan,   # flag value 0 (not used in c3s file)
                                    np.nan,   # flag value 1 (ice-free ocean)
                                    0.0,      # flag value 2 (first-year sea ice)
                                    1.0,      # flag value 3 (multi-year sea ice)
                                    0.5,      # flag value 4 (ambiguous ice type)
                                    np.nan])  # flag value 5 (used as generic fill value)
        return translator_list[flag]

    @property
    def requested_filepath(self) -> "Path":
        """
        Note: this overwrites the property in the super class due to some
        peculiarities with the filenaming (auto product changes etc)
        :return: The filepath to the target file
        """

        # The path needs to be completed if two products shall be used
        opt = self.cfg.options

        # For data records that consists of cdr/icdr only: Check if in cdr or icdr period
        # This also affects the long_name of the data set which is updated here
        is_cdr_icdr = opt.get("is_cdr_icdr", False)
        version = opt.get("version", None)
        record_type = None
        file_version = opt.get("file_version", version)
        if is_cdr_icdr:
            product_index = int(self.start_time > opt[opt.version]["cdr_time_coverage_end"])
            record_type = self.cdr_icdr_record_types[product_index]
            record_type_prefix = self.cdr_icdr_record_type_prefix[product_index]
            long_name_template = opt.get("long_name_template", {})
            long_name = long_name_template.format(record_type_prefix=record_type_prefix, version=file_version)
            self.cfg.set_long_name(long_name)

        # Get the file path
        # Paths for climate data records should contain record type and version
        path = Path(self.cfg.local_repository)
        if is_cdr_icdr:
            path = path / record_type / version

        # Add period sub-folders as indicated
        for subfolder_tag in self.cfg.subfolders:
            subfolder = getattr(self, subfolder_tag)
            path = path / subfolder

        # Construct the filename
        filename = self.cfg.filenaming.format(
            record_type=record_type,
            version=file_version,
            year=self.year,
            month=self.month,
            day=self.day,
            hemisphere_code=self.hemisphere_code)

        # Final Path
        path = path / filename
        return path

    @property
    def cdr_icdr_record_types(self) -> List[str]:
        return ["cdr", "icdr"]

    @property
    def cdr_icdr_record_type_prefix(self) -> List[str]:
        return ["", "interim"]


def fill_sitype_gaps(sitype, sitype_uncertainty, sic, sic_threshold=70., gap_filled_uncertainty=0.5,
                     max_valid_nn_dist=150, ambiguos_sitype_value=0.5):
    """
    Fill gaps in sea ice type that are indicated as sea ice in the sea ice concentration product.
    Gaps close to a valid sea-ice type value will be filled using a nearest neighbour approach,
    further gaps will be labeled as ambiguous. Any gap-filled value will be associated with a
    large uncertainty.

    :param sitype: sea-ice type array
    :param sitype_uncertainty: sea-ice type uncertainty array
    :param sic: sea-ice concentration array
    :param sic_threshold: threshold in percent as to when a waveform can be intepreted as sea ice
    :param gap_filled_uncertainty: the sea-ice type uncertainty value associated with gap filled values
    :param max_valid_nn_dist: The maximum distance (in points) for valid nearest neighbour interpolation
    :param ambiguos_sitype_value: The value associated to ambiguous gaps

    :return: Gap-filled sitype, sitype_uncertainty
    """

    # Step 1: Find gaps (if any)
    is_seaice = sic >= sic_threshold
    is_sitype_gap = np.isnan(sitype)
    is_valid_sitype = np.logical_not(is_sitype_gap)
    fillable_gap = np.logical_and(is_seaice, is_sitype_gap)
    all_gap_indices = np.where(fillable_gap)[0]

    # Check if there is anything to do
    if len(all_gap_indices) == 0:
        return sitype, sitype_uncertainty

    # Step 2: Compute the distance of a sea ice type gap to the next valid value
    # Must have at least some valid sea ice type values
    if not is_sitype_gap.all():
        gap_dist = SLABaseFunctionality().get_tiepoint_distance(is_valid_sitype)

        # Step 3: Fill closer gaps with nearest neighbour approach
        fillable_gap_nn = np.logical_and(fillable_gap, gap_dist <= max_valid_nn_dist)
        nn_gap_indices = np.where(fillable_gap_nn)[0]
        if np.count_nonzero(is_valid_sitype) > 1:
            x = np.arange(len(sitype))
            nn_interp = interpolate.interp1d(x[is_valid_sitype], sitype[is_valid_sitype],
                                             kind="nearest", fill_value="extrapolate",
                                             bounds_error=False)
            sitype[nn_gap_indices] = nn_interp(x[nn_gap_indices])

    # If no valid sea-ice type information is available than all
    # gap values exceed the maximum nearest neighbour distance
    else:
        gap_dist = np.full(sitype.shape, max_valid_nn_dist + 1)

    # Step 4: Fill rest of the gaps with ambiguous
    ambiguos_gaps = np.logical_and(fillable_gap, gap_dist > max_valid_nn_dist)
    ambiguos_gap_indices = np.where(ambiguos_gaps)[0]
    sitype[ambiguos_gap_indices] = ambiguos_sitype_value

    # Step 5: Set uncertainty of all gaps
    sitype_uncertainty[all_gap_indices] = gap_filled_uncertainty

    # Return output
    return sitype, sitype_uncertainty
