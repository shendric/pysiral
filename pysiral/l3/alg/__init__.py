import itertools
import re
import sys
import uuid
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats
from scipy.ndimage import maximum_filter
from xarray import open_dataset

from pysiral import psrlcfg
from pysiral.core.errorhandler import ErrorStatus
from pysiral.core.flags import SURFACE_TYPE_DICT, ORCondition
from pysiral.l2.alg.mask import L3Mask
from pysiral.l2.alg.sit import frb2sit_errprop
from pysiral.l3.data import L3DataGrid


class Level3ProcessorItem(object):
    """
    A parent class for processing items to be selected in the Level-3 processor settings
    and applied in the Level3Processor
    """

    def __init__(self, l3grid, **cfg):
        """
        Initizalizes the Level-3 processor item and performs checks
        if all option input parameters are available.

        :param l3grid: the Level3DataGrid instance to be processed
        :param cfg: The option dictionary/treedict from the config settings file
        """

        # Add error handler
        self.error = ErrorStatus(caller_id=self.__class__.__name__)

        # Store the arguments with type validation
        if not isinstance(l3grid, L3DataGrid):
            msg = "Invalid data type [%s] for l3grid parameter. Must be l3proc.L3DataGrid"
            msg %= type(l3grid)
            self.error.add_error("invalid-argument", msg)
            self.error.raise_on_error()
        self.l3grid = l3grid
        self.cfg = cfg

        # run the input validation checks
        self._check_variable_dependencies()
        self._check_options()

        # Add empty parameters to the l3grid
        self._add_l3_variables()

    def _check_variable_dependencies(self):
        """
        Tests if the Level-3 data grid has all required input variables (both in the Level-2 stack as
        well as in the Level 3 parameters). All processor item classes that are inheriting this class
        require the properties `l3_variable_dependencies` & `l2_variable_dependencies` for this method to work. Both
        parameter should return a list of variable names. Empty lists should be returned in case of no
        dependency.
        :return:
        """

        # Check Level-2 stack parameter
        for l2_var_name in self.l2_variable_dependencies:
            if l2_var_name not in self.l3grid.l2.stack:
                msg = "Level-3 processor item %s requires l2 stack parameter [%s], which does not exist"
                msg %= (self.__class__.__name__, l2_var_name)
                self.error.add_error("l3procitem-missing-l2stackitem", msg)
                self.error.raise_on_error()

        # Check Level-3 grid parameter
        for l3_var_name in self.l3_variable_dependencies:
            if l3_var_name not in self.l3grid.vars:
                msg = "Level-3 processor item %s requires l3 grid parameter [%s], which does not exist"
                msg %= (self.__class__.__name__, l3_var_name)
                self.error.add_error("l3procitem-missing-l3griditem", msg)
                self.error.raise_on_error()

    def _check_options(self):
        """
        Tests if the all options are given in the Level-3 processor definition files. All processor item
        classes require the property `required_options` (list of option names) for this method to work.
        NOTE: It is in the spirit of pysiral of having all numerical values in one place only that ideally
              is not the code itself.
        :return:
        """
        for option_name in self.required_options:
            option_value = self.cfg.get(option_name, None)
            if option_value is None:
                msg = "Missing option `%s` in Level-3 processor item %s" % (option_name, self.__class__.__name__)
                self.error.add_error("l3procitem-missing-option", msg)
                self.error.raise_on_error()
            setattr(self, option_name, option_value)

    def _add_l3_variables(self):
        """
        This method initializes the output variables for a given processing item to the l3grid. All processor item
        classes require the property `l3_output_variables` for this method to work. The property should return a
        dict with variable names as keys and the value a dict with fill_value and data type.
        :return:
        """
        for variable_name in self.l3_output_variables.keys():
            vardef = self.l3_output_variables[variable_name]
            self.l3grid.add_grid_variable(variable_name, vardef["fill_value"], vardef["dtype"])





class Level3ValidSeaIceFreeboardCount(Level3ProcessorItem):
    """
    A Level-3 processor item to count valid sea ice freeboard values.
    This class should be used for very limited l2i outputs files that
    do not have the surface_type variable necessary for
    `Level3SurfaceTypeStatistics`
    """

    # Mandatory properties
    required_options = []
    l2_variable_dependencies = []
    l3_variable_dependencies = ["sea_ice_freeboard"]
    l3_output_variables = dict(n_valid_freeboards=dict(dtype="i4", fill_value=0))

    def __init__(self, *args, **kwargs):
        """
        Compute surface type statistics
        :param args:
        :param kwargs:
        """
        super(Level3ValidSeaIceFreeboardCount, self).__init__(*args, **kwargs)

    def apply(self):
        """
        Computes the number of valid sea_ice_freeboard values
        """
        for xi, yj in self.l3grid.grid_indices:

            # Extract the list of surface types inm the grid cell
            sea_ice_freeboards = np.array(self.l3grid.l2.stack["sea_ice_freeboard"][yj][xi])
            self.l3grid.vars["n_valid_freeboards"][yj, xi] = np.where(np.isfinite(sea_ice_freeboards))[0].size


class Level3SurfaceTypeStatistics(Level3ProcessorItem):
    """ A Level-3 processor item to compute surface type stastics """

    # Mandatory properties
    required_options = []
    l2_variable_dependencies = ["surface_type", "sea_ice_thickness"]
    l3_variable_dependencies = []
    l3_output_variables = dict(n_total_waveforms=dict(dtype="f4", fill_value=0.0),
                               n_valid_waveforms=dict(dtype="f4", fill_value=0.0),
                               valid_fraction=dict(dtype="f4", fill_value=0.0),
                               lead_fraction=dict(dtype="f4", fill_value=0.0),
                               seaice_fraction=dict(dtype="f4", fill_value=0.0),
                               ocean_fraction=dict(dtype="f4", fill_value=0.0),
                               negative_thickness_fraction=dict(dtype="f4", fill_value=0.0),
                               is_land=dict(dtype="i2", fill_value=-1))

    def __init__(self, *args, **kwargs):
        """
        Compute surface type statistics
        :param args:
        :param kwargs:
        """
        super(Level3SurfaceTypeStatistics, self).__init__(*args, **kwargs)

        # Init this class
        self._surface_type_dict = SURFACE_TYPE_DICT

    def apply(self):
        """
        Computes the mandatory surface type statistics on the surface type stack flag

        The current list
          - is_land (land flag exists in l2i stack)
          - n_total_waveforms (size of l2i stack)
          - n_valid_waveforms (tagged as either lead, sea ice or ocean )
          - valid_fraction (n_valid/n_total)
          - lead_fraction (n_leads/n_valid)
          - ice_fraction (n_ice/n_valid)
          - ocean_fraction (n_ocean/n_valid)
          - negative thickness fraction (n_sit<0 / n_sit)
        """

        # Loop over all grid indices
        stflags = self._surface_type_dict
        for xi, yj in self.l3grid.grid_indices:

            # Extract the list of surface types inm the grid cell
            surface_type = np.array(self.l3grid.l2.stack["surface_type"][yj][xi])

            # Stack can be empty
            if len(surface_type) == 0:
                continue

            # Create a land flag
            is_land = len(np.where(surface_type == stflags["land"])[0] > 0)
            self.l3grid.vars["is_land"][xi, yj] = is_land

            # Compute total waveforms in grid cells
            n_total_waveforms = len(surface_type)
            self.l3grid.vars["n_total_waveforms"][yj, xi] = n_total_waveforms

            # Compute valid waveforms
            # Only positively identified waveforms (either lead or ice)
            valid_waveform = ORCondition()
            valid_waveform.add(surface_type == stflags["lead"])
            valid_waveform.add(surface_type == stflags["sea_ice"])
            valid_waveform.add(surface_type == stflags["ocean"])
            n_valid_waveforms = valid_waveform.num
            self.l3grid.vars["n_valid_waveforms"][yj, xi] = n_valid_waveforms

            # Fractions of leads on valid_waveforms
            try:
                valid_fraction = float(n_valid_waveforms) / float(n_total_waveforms)
            except ZeroDivisionError:
                valid_fraction = 0
            self.l3grid.vars["valid_fraction"][yj, xi] = valid_fraction

            # Fractions of surface types with respect to valid_waveforms
            for surface_type_name in ["ocean", "lead", "sea_ice"]:
                n_wfm = len(np.where(surface_type == stflags[surface_type_name])[0])
                try:
                    detection_fraction = float(n_wfm) / float(n_valid_waveforms)
                except ZeroDivisionError:
                    detection_fraction = 0
                surface_type_id = surface_type_name.replace("_", "")
                self.l3grid.vars[f"{surface_type_id}_fraction"][yj, xi] = detection_fraction

            # Fractions of negative thickness values
            sit = np.array(self.l3grid.l2.stack["sea_ice_thickness"][yj][xi])
            n_negative_thicknesses = len(np.where(sit < 0.0)[0])
            try:
                n_ice = len(np.where(surface_type == stflags["sea_ice"])[0])
                negative_thickness_fraction = float(n_negative_thicknesses) / float(n_ice)
            except ZeroDivisionError:
                negative_thickness_fraction = np.nan
            self.l3grid.vars["negative_thickness_fraction"][yj, xi] = negative_thickness_fraction


class Level3TemporalCoverageStatistics(Level3ProcessorItem):
    """
    A Level-3 processor item to compute temporal coverage statistics of sea-ice thickness in the grid period
    """

    # Mandatory properties
    required_options = []
    l2_variable_dependencies = ["time", "sea_ice_thickness"]
    l3_variable_dependencies = []
    l3_output_variables = dict(temporal_coverage_uniformity_factor=dict(dtype="f4", fill_value=np.nan),
                               temporal_coverage_day_fraction=dict(dtype="f4", fill_value=np.nan),
                               temporal_coverage_period_fraction=dict(dtype="f4", fill_value=np.nan),
                               temporal_coverage_weighted_center=dict(dtype="f4", fill_value=np.nan))

    def __init__(self, *args, **kwargs):
        """
        Compute surface type statistics
        :param args:
        :param kwargs:
        """
        super(Level3TemporalCoverageStatistics, self).__init__(*args, **kwargs)

    def apply(self):
        """
        Computes statistics of the temporal coverage of sea ice thickness
        :return:
        """

        # Other parameter for L3DataGrid
        # All statistics are computed with respect to the temporal coverage of the grid
        # (-> the period that has been asked for, not the actual data coverage)
        # TODO: TCS & TCE as datetime should not be in the metadata, but a property of the data object
        tcs, tce = self.l3grid.metadata.time_coverage_start, self.l3grid.metadata.time_coverage_end
        start_date = date(tcs.year, tcs.month, tcs.day)
        end_date = date(tce.year, tce.month, tce.day)
        period_n_days = (end_date - start_date).days + 1

        # Links
        stack = self.l3grid.l2.stack

        # Loop over all grid cells
        for xi, yj in self.l3grid.grid_indices:

            # Get the day of observation for each entry in the Level-2 stack
            times = np.array(stack["time"][yj][xi])
            day_of_observation = np.array([date(t.year, t.month, t.day) for t in times])

            # The statistic is computed for sea ice thickness -> remove data points without valid sea ice thickness
            sea_ice_thickness = np.array(stack["sea_ice_thickness"][yj][xi])
            day_of_observation = day_of_observation[np.isfinite(sea_ice_thickness)]

            # Validity check
            #  - must have data
            if len(day_of_observation) == 0:
                continue

            # Compute the number of days for each observation with respect to the start of the period
            day_number = [(day - start_date).days for day in day_of_observation]

            # Compute the set of days with observations available
            days_with_observations = np.unique(day_number)
            first_day, last_day = np.amin(days_with_observations), np.amax(days_with_observations)

            # Compute the uniformity factor
            # The uniformity factor is derived from a Kolmogorov-Smirnov (KS) test for goodness of fit that tests
            # the list of against a uniform distribution. The definition of the uniformity factor is that is
            # reaches 1 for uniform distribution of observations and gets smaller for non-uniform distributions
            # It is therefore defined as 1-D with D being the result of KS test
            ks_test_result = stats.kstest(day_number, stats.uniform(loc=0.0, scale=period_n_days).cdf)
            uniformity_factor = 1.0 - ks_test_result[0]
            self.l3grid.vars["temporal_coverage_uniformity_factor"][yj, xi] = uniformity_factor

            # Compute the day fraction (number of days with actual data coverage/days of period)
            day_fraction = float(len(days_with_observations)) / float(period_n_days)
            self.l3grid.vars["temporal_coverage_day_fraction"][yj, xi] = day_fraction

            # Compute the period in days that is covered between the first and last day of observation
            # normed by the length of the period
            period_fraction = float(last_day - first_day + 1) / float(period_n_days)
            self.l3grid.vars["temporal_coverage_period_fraction"][yj, xi] = period_fraction

            # Compute the temporal center of the actual data coverage in units of period length
            # -> optimum 0.5
            weighted_center = np.mean(day_number) / float(period_n_days)
            self.l3grid.vars["temporal_coverage_weighted_center"][yj, xi] = weighted_center


class Level3StatusFlag(Level3ProcessorItem):
    """
    A Level-3 processor item to compute the status flag
    """

    # Mandatory properties
    required_options = ["retrieval_status_target", "sic_thrs", "flag_values"]
    l2_variable_dependencies = []
    l3_variable_dependencies = ["sea_ice_concentration", "n_valid_waveforms", "landsea"]
    l3_output_variables = dict(status_flag=dict(dtype="i1", fill_value=1))

    def __init__(self, *args, **kwargs):
        """
        Compute surface type statistics
        :param args:
        :param kwargs:
        """
        super(Level3StatusFlag, self).__init__(*args, **kwargs)

    def apply(self):
        """
        Computes the status flag
        :return:
        """

        # Get the flag values from the l3 settings file
        flag_values = self.flag_values

        # Get status flag (fill value should be set to zero)
        sf = np.copy(self.l3grid.vars["status_flag"])

        # Init the flag with not data flag value
        sf[:] = flag_values["no_data"]

        # get input parameters
        par = np.copy(self.l3grid.vars[self.retrieval_status_target])
        sic = self.l3grid.vars["sea_ice_concentration"]
        nvw = self.l3grid.vars["n_valid_waveforms"]
        lnd = self.l3grid.vars["landsea"]

        # --- Compute conditions for flags ---

        # Get sea ice mask
        is_below_sic_thrs = np.logical_and(sic >= 0., sic < self.sic_thrs)

        # Get the pole hole information
        mission_ids = self.l3grid.metadata.mission_ids.split(",")
        orbit_inclinations = [psrlcfg.platforms.get_orbit_inclination(mission_id) for mission_id in mission_ids]

        # NOTE: due to varying grid cell size, it is no sufficient to just check which grid cell coordinate
        #       is outside the orbit coverage
        is_pole_hole = np.logical_and(
            np.abs(self.l3grid.vars["latitude"]) > (np.amin(orbit_inclinations)-1.0),
            flag_values["no_data"])

        # Check where the retrieval has failed
        is_land = lnd > 0
        has_data = nvw > 0
        has_retrieval = np.isfinite(par)
        retrieval_failed = np.logical_and(
            np.logical_and(has_data, np.logical_not(is_below_sic_thrs)),
            np.logical_not(has_retrieval))

        # Set sic threshold
        sf[np.where(is_below_sic_thrs)] = flag_values["is_below_sic_thrs"]

        # Set pole hole (Antarctica: Will be overwritten below)
        sf[np.where(is_pole_hole)] = flag_values["is_pole_hole"]

        # Set land mask
        sf[np.where(is_land)] = flag_values["is_land"]

        # Set failed retrieval
        sf[np.where(retrieval_failed)] = flag_values["retrieval_failed"]

        # Set retrieval successful
        sf[np.where(has_retrieval)] = flag_values["has_retrieval"]

        # Write Status flag
        self.l3grid.vars["status_flag"] = sf


class Level3QualityFlag(Level3ProcessorItem):
    """
    A Level-3 processor item to compute the status flag
    """

    # Mandatory properties
    required_options = ["add_rule_flags", "rules"]
    l2_variable_dependencies = []
    l3_variable_dependencies = ["sea_ice_thickness", "n_valid_waveforms", "negative_thickness_fraction",
                                "lead_fraction"]
    l3_output_variables = dict(quality_flag=dict(dtype="i1", fill_value=3))

    def __init__(self, *args, **kwargs):
        """
        Compute surface type statistics
        :param args:
        :param kwargs:
        """
        super(Level3QualityFlag, self).__init__(*args, **kwargs)

    def apply(self):
        """ Computation of quality flag indicator based on several rules defined in the l3 settings file """

        # Get the quality flag indicator array
        # This array will be continously updated by the quality check rules
        qif = np.copy(self.l3grid.vars["quality_flag"])
        sit = np.copy(self.l3grid.vars["sea_ice_thickness"])
        nvw = np.copy(self.l3grid.vars["n_valid_waveforms"])
        ntf = np.copy(self.l3grid.vars["negative_thickness_fraction"])
        lfr = np.copy(self.l3grid.vars["lead_fraction"])

        # As first step set qif to 1 where data is availabe
        qif[np.where(np.isfinite(sit))] = 0

        # Get a list of all the rules
        quality_flag_rules = self.rules.keys()

        # Simple way of handling rules (for now)

        # Use the Warren99 validity maslk
        # XXX: Not implemented yet
        if "qif_warren99_valid_flag" in quality_flag_rules:
            w99 = self.l3grid.vars["warren99_is_valid"]
            # mask = 0 means warren99 is invalid
            rule_options = self.rules["qif_warren99_valid_flag"]
            flag = np.full(qif.shape, 0, dtype=qif.dtype)
            flag[np.where(w99 == 0)] = rule_options["target_flag"]
            qif = np.maximum(qif, flag)

        # Elevate the quality flag for SARin or mixed SAR/SARin regions
        # (only sensible for CryoSat-2)
        if "qif_cs2_radar_mode_is_sin" in quality_flag_rules:
            radar_modes = self.l3grid.vars["radar_mode"]
            rule_options = self.rules["qif_cs2_radar_mode_is_sin"]
            flag = np.full(qif.shape, 0, dtype=qif.dtype)
            flag[np.where(radar_modes >= 2.)] = rule_options["target_flag"]
            qif = np.maximum(qif, flag)

        # Check the number of waveforms (less valid waveforms -> higher warning flag)
        if "qif_n_waveforms" in quality_flag_rules:
            flag = np.full(qif.shape, 0, dtype=qif.dtype)
            rule_options = self.rules["qif_n_waveforms"]
            for threshold, target_flag in zip(rule_options["thresholds"], rule_options["target_flags"]):
                flag[np.where(nvw < threshold)] = target_flag
            qif = np.maximum(qif, flag)

        # Check the availiability of leads in an area adjacent to the grid cell
        if "qif_lead_availability" in quality_flag_rules:
            flag = np.full(qif.shape, 0, dtype=qif.dtype)
            rule_options = self.rules["qif_lead_availability"]
            # get the window size
            grid_res = self.l3grid.griddef.resolution
            window_size = np.ceil(rule_options["search_radius_m"] / grid_res)
            window_size = int(2 * window_size + 1)
            # Use a maximum filter to get best lead fraction in area
            area_lfr = maximum_filter(lfr, size=window_size)
            thrs = rule_options["area_lead_fraction_minimum"]
            flag[np.where(area_lfr <= thrs)] = rule_options["target_flag"]
            qif = np.maximum(qif, flag)

        if "qif_miz_flag" in quality_flag_rules:
            flag = np.full(qif.shape, 0, dtype=qif.dtype)
            rule_options = self.rules["qif_miz_flag"]
            for source_flag, target_flag in zip(rule_options["source_flags"], rule_options["source_flags"]):
                flag[np.where(self.l3grid.vars["flag_miz"] == source_flag)] = target_flag
            qif = np.maximum(qif, flag)

        # Check the negative thickness fraction (higher value -> higher warnung flag)
        if "qif_high_negative_thickness_fraction" in quality_flag_rules:
            flag = np.full(qif.shape, 0, dtype=qif.dtype)
            rule_options = self.rules["qif_high_negative_thickness_fraction"]
            for threshold, target_flag in zip(rule_options["thresholds"], rule_options["target_flags"]):
                flag[np.where(ntf > threshold)] = target_flag
            qif = np.maximum(qif, flag)

        # Set all flags with no data to last flag value again
        qif[np.where(np.isnan(sit))] = 3

        # Set flag again
        self.l3grid.vars["quality_flag"] = qif


class Level3LoadMasks(Level3ProcessorItem):
    """
    A Level-3 processor item to load external masks
    """

    # Mandatory properties
    required_options = ["mask_names"]
    l2_variable_dependencies = []
    l3_variable_dependencies = []
    # Note: the output names depend on mask name, thus these will be
    #       created in apply (works as well)
    l3_output_variables = dict()

    def __init__(self, *args, **kwargs):
        """
        Compute surface type statistics
        :param args:
        :param kwargs:
        """
        super(Level3LoadMasks, self).__init__(*args, **kwargs)

    def apply(self):
        """
        Load masks and add them as grid variable (variable name -> mask name)
        :return:
        """

        # Get the mask names and load each
        for mask_name in self.mask_names:

            # The masks are stored in external files that can be automatically
            #  found with the grid id
            mask = L3Mask(mask_name, self.l3grid.griddef.grid_id)

            # Add the mask to the l3grid as variable
            if not mask.error.status:
                self.l3grid.add_grid_variable(mask_name, np.nan, mask.mask.dtype)
                self.l3grid.vars[mask_name] = mask.mask

            # If fails, only add an empty variable
            else:
                self.l3grid.add_grid_variable(mask_name, np.nan, "f4")
                error_msgs = mask.error.get_all_messages()
                for error_msg in error_msgs:
                    logger.error(error_msg)


class Level3LoadCCILandMask(Level3ProcessorItem):
    """
    A Level-3 processor item to load the CCI land mask
    """

    # Mandatory properties
    required_options = ["local_machine_def_mask_tag", "mask_name_dict"]
    l2_variable_dependencies = []
    l3_variable_dependencies = []
    # Note: the output names depend on mask name, thus these will be
    #       created in apply (works as well)
    l3_output_variables = dict()

    def __init__(self, *args, **kwargs):
        """
        Initiate the class
        :param args:
        :param kwargs:
        """
        super(Level3LoadCCILandMask, self).__init__(*args, **kwargs)

    def apply(self):
        """
        Load masks and add them as grid variable (variable name -> mask name)
        :return:
        """

        # Short cut
        grid_id = self.l3grid.griddef.grid_id

        # Get mask target path:
        mask_tag = self.cfg["local_machine_def_mask_tag"]
        lookup_directory = psrlcfg.local_machine.auxdata_repository.mask.get(mask_tag, None)
        if lookup_directory is None:
            msg = "Missing local machine def tag: auxdata_repository.mask.{}".format(mask_tag)
            self.error.add_error("invalid-local-machine-def", msg)
            logger.error(msg)
            return
        lookup_directory = Path(lookup_directory)

        # Get the mask target filename
        filename = self.cfg["mask_name_dict"][grid_id.replace("_", "")]
        mask_filepath = lookup_directory / filename
        if not mask_filepath.is_file():
            msg = "Missing input file: {}".format(mask_filepath)
            self.error.add_error("invalid-local-machine-def", msg)
            logger.error(msg)
            return

        # Load the data and extract the flag
        nc = open_dataset(str(mask_filepath), decode_times=False)

        # The target land sea flag should be 1 for land and 0 for sea,
        # but the CCI landsea mask provides also fractional values for
        # mixed surfaces types. Thus, we add two arrays to the
        # L3grid object
        #   1. a classical land/sea mask with land:1 and sea: 0. In
        #      this notation the mixed pixels are attributed to sea
        #      because there might be some valid retrieval there
        #   2. the ocean density value as is
        density_of_ocean = np.flipud(nc.density_of_ocean.values)
        landsea_mask = density_of_ocean < 1e-5

        # Add mask to l3 grid
        mask_variable_name = self.cfg["mask_variable_name"]
        self.l3grid.add_grid_variable(mask_variable_name, np.nan, landsea_mask.dtype)
        self.l3grid.vars[mask_variable_name] = landsea_mask.astype(int)

        density_variable_name = self.cfg["density_variable_name"]
        self.l3grid.add_grid_variable(density_variable_name, np.nan, nc.density_of_ocean.values.dtype)
        self.l3grid.vars[density_variable_name] = density_of_ocean


class Level3GridUncertainties(Level3ProcessorItem):
    """
    A Level-3 processor item to compute uncertainties of key geophysical variables on a grid.
    NOTE: As a concession to backward compability: sea ice draft uncertainty will be computed, but
          the sea ice draft is not a required input parameter
    """

    # Mandatory properties
    required_options = ["water_density", "snow_depth_correction_factor", "max_l3_uncertainty"]
    l2_variable_dependencies = ["radar_freeboard_uncertainty", "sea_ice_thickness"]
    l3_variable_dependencies = ["sea_ice_thickness", "sea_ice_freeboard", "snow_depth", "sea_ice_density",
                                "snow_density", "snow_depth_uncertainty", "sea_ice_density_uncertainty",
                                "snow_density_uncertainty"]
    l3_output_variables = dict(radar_freeboard_l3_uncertainty=dict(dtype="f4", fill_value=np.nan),
                               freeboard_l3_uncertainty=dict(dtype="f4", fill_value=np.nan),
                               sea_ice_thickness_l3_uncertainty=dict(dtype="f4", fill_value=np.nan),
                               sea_ice_draft_l3_uncertainty=dict(dtype="f4", fill_value=np.nan))

    def __init__(self, *args, **kwargs):
        """
        Compute surface type statistics
        :param args:
        :param kwargs:
        """
        super(Level3GridUncertainties, self).__init__(*args, **kwargs)

    def apply(self):
        """ Compute a level 3 uncertainty. The general idea is to compute the error propagation of average
        error components, where for components for random error the error of the l2 average
        is used and for systematic error components the average of the l2 error """

        # Options
        rho_w = self.water_density
        sd_corr_fact = self.snow_depth_correction_factor

        # Loop over grid items
        for xi, yj in self.l3grid.grid_indices:

            # Check of data exists
            if np.isnan(self.l3grid.vars["sea_ice_thickness"][yj, xi]):
                continue

            # Get parameters
            frb = self.l3grid.vars["sea_ice_freeboard"][yj, xi]
            sd = self.l3grid.vars["snow_depth"][yj, xi]
            rho_i = self.l3grid.vars["sea_ice_density"][yj, xi]
            rho_s = self.l3grid.vars["snow_density"][yj, xi]

            # Get systematic error components
            sd_unc = self.l3grid.vars["snow_depth_uncertainty"][yj, xi]
            rho_i_unc = self.l3grid.vars["sea_ice_density_uncertainty"][yj, xi]
            rho_s_unc = self.l3grid.vars["snow_density_uncertainty"][yj, xi]

            # Get random uncertainty
            # Note: this applies only to the radar freeboard uncertainty.
            #       Thus we need to recalculate the sea ice freeboard uncertainty

            # Get the stack of radar freeboard uncertainty values and remove NaN's
            # rfrb_unc = self.l3["radar_freeboard_uncertainty"][yj, xi]
            rfrb_uncs = np.array(self.l3grid.l2.stack["radar_freeboard_uncertainty"][yj][xi])
            rfrb_uncs = rfrb_uncs[~np.isnan(rfrb_uncs)]

            # Compute radar freeboard uncertainty as error or the mean from values with individual
            # error components (error of a weighted mean)
            weight = np.nansum(1. / rfrb_uncs ** 2.)
            rfrb_unc = 1. / np.sqrt(weight)
            self.l3grid.vars["radar_freeboard_l3_uncertainty"][yj, xi] = rfrb_unc

            # Calculate the level-3 freeboard uncertainty with updated radar freeboard uncertainty
            deriv_snow = sd_corr_fact
            frb_unc = np.sqrt((deriv_snow * sd_unc) ** 2. + rfrb_unc ** 2.)
            self.l3grid.vars["freeboard_l3_uncertainty"][yj, xi] = frb_unc

            # Calculate the level-3 thickness uncertainty
            errprop_args = [frb, sd, rho_w, rho_i, rho_s, frb_unc, sd_unc, rho_i_unc, rho_s_unc]
            sit_l3_unc = frb2sit_errprop(*errprop_args)

            # Cap the uncertainty
            # (very large values may appear in extreme cases)
            if sit_l3_unc > self.max_l3_uncertainty:
                sit_l3_unc = self.max_l3_uncertainty

            # Assign Level-3 uncertainty
            self.l3grid.vars["sea_ice_thickness_l3_uncertainty"][yj, xi] = sit_l3_unc

            # Compute sea ice draft uncertainty
            if "sea_ice_draft" not in self.l3grid.vars:
                continue

            sid_l3_unc = np.sqrt(sit_l3_unc ** 2. + frb_unc ** 2.)
            self.l3grid.vars["sea_ice_draft_l3_uncertainty"][yj, xi] = sid_l3_unc


class Level3ParameterMask(Level3ProcessorItem):
    """
    A Level-3 processor item to load external masks
    """

    # Mandatory properties
    required_options = ["source", "condition", "targets"]
    l2_variable_dependencies = []
    l3_variable_dependencies = []
    # Note: the output names depend on mask name, thus these will be
    #       created in apply (works as well)
    l3_output_variables = dict()

    def __init__(self, *args, **kwargs):
        """
        Compute surface type statistics
        :param args:
        :param kwargs:
        """
        super(Level3ParameterMask, self).__init__(*args, **kwargs)

    def apply(self):
        """
        Mask certain parameters based on condition of one other parameter
        :return:
        """

        # Get the source parameter
        source = self.l3grid.vars[self.source]

        # Compute the masking condition
        conditions = self.condition.split(";")
        n_conditions = len(conditions)

        if n_conditions == 0:
            msg = "Missing condition in %s" % self.__class__.__name__
            self.error.add_error("invalid-l3mask-def", msg)
            return

        # Start with the first (and maybe only condition)
        filter_mask = self._get_l3_mask(source, conditions[0], self.cfg)

        # Add conditions
        if n_conditions >= 2:
            for i in range(1, n_conditions):
                new_filter = self._get_l3_mask(source, conditions[i], self.cfg)
                if self.cfg["connect_conditions"] == "or":
                    filter_mask = np.logical_or(filter_mask, new_filter)
                elif self.cfg["connect_conditions"] == "and":
                    filter_mask = np.logical_and(filter_mask, new_filter)
                else:
                    msg = "Invalid l3 mask operation: %s"
                    msg %= self.cfg["connect_conditions"]
                    self.error.add_error("invalid-l3mask-def", msg)
                    self.error.raise_on_error()

        # Apply mask
        masked_indices = np.where(filter_mask)
        for target in self.targets:
            try:
                self.l3grid.vars[target][masked_indices] = np.nan
            except ValueError:
                if self.l3grid.vars[target].dtype.kind == "i":
                    self.l3grid.vars[target][masked_indices] = -1
                else:
                    msg = "Cannot set nan (or -1) as mask value to parameter: %s " % target
                    logger.warning(msg)

    def _get_l3_mask(self, source_param, condition, options):
        """ Return bool array based on a parameter and a predefined
        masking operation """
        if condition.strip() == "is_nan":
            return np.isnan(source_param)
        elif condition.strip() == "is_zero":
            return np.array(source_param <= 1.0e-9)
        elif condition.strip() == "is_smaller":
            return np.array(source_param < options["is_smaller_threshold"])
        else:
            msg = "Unknown condition in l3 mask: %s" % condition
            self.error.add_error("invalid-l3mask-condition", msg)
            self.error.raise_on_error()


class Level3GriddedClassifiers(Level3ProcessorItem):
    """
    A Level-3 processor item to provide gridded classifiers (for different surface types)
    """

    # Mandatory properties
    required_options = ["parameters", "surface_types", "statistics"]
    l2_variable_dependencies = ["surface_type"]
    l3_variable_dependencies = []
    # Note: the output names depend on the parameters selected, thus these will be
    #       created in apply (works as well)
    l3_output_variables = dict()

    def __init__(self, *args, **kwargs):
        """
        Compute surface type statistics
        :param args:
        :param kwargs:
        """
        super(Level3GriddedClassifiers, self).__init__(*args, **kwargs)

        # Surface type dict is required to get subsets
        self._surface_type_dict = SURFACE_TYPE_DICT

        # Statistical function dictionary
        self._stat_functions = dict(mean=lambda x: np.nanmean(x), sdev=lambda x: np.nanstd(x))

    def apply(self):
        """
        Mask certain parameters based on condition of one other parameter
        :return:
        """

        # Get surface type flag
        surface_type = self.l3grid.l2.stack["surface_type"]
        target_surface_types = list(self.surface_types)
        target_surface_types.append("all")

        # Loop over all parameters
        for parameter_name in self.parameters:
            # Get the stack
            classifier_stack = None
            try:
                classifier_stack = self.l3grid.l2.stack[parameter_name]
            except KeyError:
                msg = "Level-3 processor item %s requires l2 stack parameter [%s], which does not exist"
                msg %= (self.__class__.__name__, parameter_name)
                self.error.add_error("l3procitem-missing-l2stackitem", msg)
                self.error.raise_on_error()

            # Loop over the statistical parameters (mean, sdev, ...)
            for statistic in self.statistics:
                # Loop over target surface types
                for target_surface_type in target_surface_types:
                    self._compute_grid_variable(parameter_name,
                                                classifier_stack,
                                                surface_type,
                                                target_surface_type,
                                                statistic)

    def _compute_grid_variable(self, parameter_name, classifier_stack, surface_type, target_surface_type, statistic):
        """
        Computes gridded surface type statistics for all grid cells
        :param parameter_name: The name of the classifier (for output name generation)
        :param classifier_stack: The Level-2 stack for the given classifier
        :param surface_type: The Level-2 stack of surface type
        :param target_surface_type: The name of the target surface type
        :param statistic: The name of the statistic to be computed
        :return:
        """

        # Create the output parameter name
        grid_var_name = "stat_%s_%s_%s" % (parameter_name, target_surface_type, statistic)
        self.l3grid.add_grid_variable(grid_var_name, np.nan, "f4")

        # Loop over all grid cells
        for xi, yj in self.l3grid.grid_indices:

            classifier_grid_values = np.array(classifier_stack[yj][xi])
            surface_type_flags = np.array(surface_type[yj][xi])

            # Get the surface type target subset
            if target_surface_type == "all":
                subset = np.arange(len(classifier_grid_values))
            else:
                try:
                    surface_type_target_flag = self._surface_type_dict[target_surface_type]
                    subset = np.where(surface_type_flags == surface_type_target_flag)[0]
                except KeyError:
                    msg = "Surface type %s does not exist" % target_surface_type
                    self.error.add_error("l3procitem-incorrect-option", msg)
                    self.error.raise_on_error()

            # A minimum of two values is needed to compute statistics
            if len(subset) < 2:
                continue
            result = self._stat_functions[statistic](classifier_grid_values[subset])
            self.l3grid.vars[grid_var_name][yj][xi] = result
