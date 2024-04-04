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

from pysiral import __version__, get_cls, psrlcfg
from pysiral.core.config import get_yaml_as_dict
from pysiral.core.errorhandler import ErrorStatus
from pysiral.core.flags import SURFACE_TYPE_DICT, ORCondition
from pysiral.core.output import Level3Output, OutputHandlerBase
from pysiral.l2 import L2iNCFileImport
from pysiral.mask import L3Mask
from pysiral.l2.alg.sit import frb2sit_errprop


class L2iDataStack(object):

    def __init__(self, griddef, l2_parameter):
        """
        A container for stacking l2i variables (geophysical parameter at sensor resolution)
        in L3 grid cells. For each parameter a (numx, numy) array is created, with a
        list containing all l2i data points that fall into the grid cell area.
        This list can be averaged or used for histogram computation in later stages
        of the Level-3 processor.

        Args:
            griddef (obj): pysiral.grid.GridDefinition or inheritated objects
            l2_parameter (str list): list of l2i parameter names

        Returns:
            class instance

        """

        # Grid Definition Type
        self.griddef = griddef

        # A list of level-2 parameters to be stacked
        self.l2_parameter = l2_parameter

        # Statistics
        self._n_records = 0
        self._l2i_count = 0
        self.start_time = []
        self.stop_time = []
        self.mission = []
        self.timeliness = []

        # Save global attributes from l2i (will be overwritten for each
        # l2i file, but it is assumed that general information, e.g.
        # on auxdata remains the same)
        self._l2i_info = None

        # Create parameter stacks
        self._initialize_stacks()

    def _initialize_stacks(self):
        """ Create all data stacks, content will be added sequentially
        with `add` method """

        # Stack dictionary that will hold the data
        self.stack = {
            parameter_name: self.parameter_stack
            for parameter_name in self.l2_parameter.keys()
        }

    def add(self, l2i):
        """ Add a l2i data object to the stack

        Args:
            l2i (obj): l2i object (currently: pysiral.l2data.L2iNCFileImport)

        Returns:
            None
        """

        # Save the metadata from the orbit data
        time = l2i.time if hasattr(l2i, "time") else l2i.timestamp
        self.start_time.append(time[0])
        self.stop_time.append(time[-1])
        self.mission.append(l2i.mission)
        self.timeliness.append(l2i.timeliness)
        self._l2i_count += 1
        self._n_records += l2i.n_records

        self._l2i_info = l2i.info

        # Get projection coordinates for l2i locations
        xi, yj = self.griddef.grid_indices(l2i.longitude, l2i.latitude)

        # Stack the l2 parameter in the corresponding grid cells
        outside_grid_flag = False
        for i in np.arange(l2i.n_records):

            # Add the surface type per default
            # (will not be gridded, therefore not in list of l2 parameter)
            x, y = int(xi[i]), int(yj[i])
            for parameter_name in self.l2_parameter.keys():
                try:
                    data = getattr(l2i, parameter_name)
                    self.stack[parameter_name][y][x].append(data[i])
                except AttributeError:
                    pass
                except IndexError:
                    outside_grid_flag = True

        if outside_grid_flag:
            logger.warning("L2 input data outside grid definition")

    @property
    def n_total_records(self):
        return int(self._n_records)

    @property
    def l2i_count(self):
        return int(self._l2i_count)

    @property
    def parameter_stack(self):
        dimx, dimy = self.griddef.extent.numx, self.griddef.extent.numy
        return [[[] for _ in range(dimx)] for _ in range(dimy)]

    @property
    def l2i_info(self):
        return self._l2i_info


class L3DataGrid(object):
    """
    Container for computing gridded data sets based on a l2i data stack
    (averaged l2i parameter, grid cell statistics)
    """

    def __init__(self, job, stack, period, doi=""):

        self.error = ErrorStatus(caller_id=self.__class__.__name__)

        # Grid size definition
        self._doi = doi
        self._data_record_type = "none"
        self._griddef = job.grid
        self._l3def = job.l3def
        self._period = period
        self._external_masks = {}

        # Check if any output definition requires an unlimited dimension
        # XXX: This is an unsatisfying solution resulting from the current solution that the L3DataGrid
        #      provides fix dimensions for potentially more than one output handler. This need
        #      consideration in the future
        self._time_dim_is_unlimited = [handler.time_dim_is_unlimited for handler in job.outputs]

        # Define time of dataset creation as the time of object initialization
        # to avoid slightly different timestamps for repeated calls of
        # datetime.now()
        self._creation_time = datetime.now()

        # # Shortcut to the surface type flag dictionary
        # self._surface_type_dict = SurfaceType.SURFACE_TYPE_DICT

        # List of level-2 parameter
        # (gridded parameter that are already in l2i)
        self._l2_parameter = None

        # list of stacked l2 parameters for each grid cell
        if not isinstance(stack, L2iDataStack):
            msg = "Input must be of type pysiral.l3proc.L2DataStack, was %s"
            msg %= type(stack)
            raise ValueError(msg)
        self.l2 = stack

        # Get a list of non-empty
        self._non_empty_grid_indices = None
        self._init_grid_indices_mask()

        # container for gridded parameters
        self.vars = {}

        # Product Metadata
        self._metadata = None
        self._init_metadata_from_l2()

        # Compute the longitude & latitude dimensions
        self.calculate_longitude_latitude_fields()

        # Create the parameter fields
        self._init_parameter_fields(job.l2_parameter)

        # Grid the Level-2 parameter
        logger.info("Grid Level-2 parameter")
        self.grid_l2_parameter()

    def set_doi(self, doi):
        # TODO: Move to __init__
        self._doi = doi

    def set_data_record_type(self, data_record_type):
        self._data_record_type = data_record_type

    def get_attribute(self, attribute_name, *args):
        """ Return a string for a given attribute name. This method is
        required for the output data handler """
        try:
            attr_getter = getattr(self, "_get_attr_" + attribute_name)
            return attr_getter(*args)
        except AttributeError:
            return "attr_unavailable"
        except Exception as ex:
            msg = "L3DataGrid.get_attribute Exception: " + str(ex) + " for attribute: %s" % attribute_name
            logger.error(msg)
            return "unknown"

    def add_grid_variable(self, parameter_name, fill_value, dtype):
        """
        Add a grid variable and fill with empty values
        :param parameter_name: The name of the parameter
        :param fill_value: the "empty" value assigned to all cell
        :param dtype: numpy compatible dtype
        :return:
        """

        # Check if variable already exists
        if parameter_name in self.vars:
            msg = "Variable overwrite alert: %s" % parameter_name
            self.error.add_error("l3-variable-overwrite", msg)
            self.error.raise_on_error()

        # All clear, create variable
        self.vars[parameter_name] = np.full(self.grid_shape, fill_value, dtype=dtype)

        # Log
        logger.info("Added grid parameter: %s" % parameter_name)

    def calculate_longitude_latitude_fields(self):
        """ Geographic coordinates from GridDefinition """
        lon, lat = self.griddef.get_grid_coordinates()
        self.vars["longitude"] = lon
        self.vars["latitude"] = lat

    def grid_l2_parameter(self):
        """ Compute averages of all l2i parameter for each grid cell.
        The list of l2i parameter is from the output format definition
        No averages are computed for grid cells that are tagged with
        a land flag. """

        settings = self.l3def.grid_settings

        # Loop over all parameter / grid cells
        for name in self._l2_parameter:

            # Certain parameters in the l2 stack are excluded from gridding
            # (-> indicated by grid_method: none)
            grid_method = self.l3def.l2_parameter[name]["grid_method"]
            if grid_method == "none":
                continue

            logger.info("Gridding parameter: %s [%s]" % (name, grid_method))

            for xi, yj in self.grid_indices:

                data = np.array(self.l2.stack[name][yj][xi])

                # nanmean needs at least 2 valid items
                valid = np.where(np.isfinite(data))[0]
                if len(valid) < settings.minimum_valid_grid_points:
                    continue

                # TODO: Think of a dicts with lambdas to make this more concise
                if grid_method == "average":
                    self.vars[name][yj, xi] = np.nanmean(data)
                elif grid_method == "average_uncertainty":
                    value = np.abs(np.sqrt(1. / np.sum(data[valid])))
                    self.vars[name][yj, xi] = value
                elif grid_method == "unique":
                    self.vars[name][yj, xi] = np.unique(data)
                elif grid_method == "median":
                    self.vars[name][yj, xi] = np.nanmedian(data)
                else:
                    msg = "Invalid grid method (%s) for %s"
                    msg %= (str(grid_method), name)
                    self.error.add_error("invalid-l3def", msg)
                    self.error.raise_on_error()

    def get_parameter_by_name(self, name, raise_on_error=True):
        try:
            parameter = self.vars[name]
        except KeyError:
            parameter = np.full(np.shape(self.vars["longitude"]), np.nan)
            logger.warning("Parameter not available: %s" % name)
        except Exception as ex:
            msg = "L3DataGrid.get_parameter_by_name Exception: " + str(ex)
            logger.error(msg)
            # TODO: use error handler
            if raise_on_error:
                sys.exit(1)
            else:
                parameter = np.full(np.shape(self.vars["longitude"]), np.nan)

        return parameter

    def set_parameter_by_name(self, name, var):
        try:
            self.vars[name] = var
        except KeyError:
            logger.warning("Parameter not available: %s" % name)
        except Exception as ex:
            print("L3DataGrid.get_parameter_by_name Exception: " + str(ex))
            sys.exit(1)

    def _init_grid_indices_mask(self) -> None:
        """
        Compute a mask of non-empty grid indices
        :return:
        """

        # Get number of items per stack
        n_records = np.ndarray(shape=self.grid_shape)
        for xi, yj in self.all_grid_indices:
            n_records[yj][xi] = len(self.l2.stack["time"][yj][xi])

        # Get indices
        self._non_empty_grid_indices = np.flip(np.array(np.where(n_records > 0))).T

    def _init_metadata_from_l2(self):
        """
        Gets metadata from Level-2 instance
        :return:
        """
        # Get the metadata information from the L2 stack
        logger.info("Compile metadata")
        self._metadata = L3MetaData()
        self._metadata.get_missions_from_stack(self.l2)
        # Actual data coverage
        self._metadata.get_data_period_from_stack(self.l2)
        # Requested time coverage (might not be the actual coverage)
        self._metadata.get_time_coverage_from_period(self._period)
        try:
            self._metadata.get_auxdata_infos(self.l2.l2i_info)
        except AttributeError:
            pass
        self._metadata.get_projection_parameter(self._griddef)

    def _init_parameter_fields(self, pardefs):
        """ Initialize output parameter fields """
        # Store the name of the parameters
        self._l2_parameter = sorted(pardefs.keys())
        for parameter_name, pardef in list(pardefs.items()):
            fillvalue = pardef["fillvalue"]
            if pardef["grid_method"] != "none":
                self.add_grid_variable(parameter_name, fillvalue, pardef["dtype"])

    def _get_attr_source_mission_id(self, *args):
        mission_ids = self.metadata.mission_ids
        if args[0] == "uppercase":
            mission_ids = mission_ids.upper()
        return mission_ids

    def _get_attr_source_mission_name(self, *args):
        ids = self.metadata.mission_ids
        return ",".join([psrlcfg.platforms.get_name(m) for m in ids.split(",")])

    def _get_attr_source_timeliness(self, *args):
        timeliness = self.metadata.source_timeliness
        if args[0] == "lowercase":
            timeliness = timeliness.lower()
        elif args[0] == "select":
            choices = {"nrt": args[1], "rep": args[2], "ntc": args[2]}
            return choices.get(timeliness.lower(), "n/a")
        return timeliness

    def _get_attr_grid_id(self, *args):
        grid_id = self.griddef.grid_id
        if args[0] == "uppercase":
            grid_id = grid_id.upper()
        return grid_id

    def _get_attr_grid_spacing_tag(self, *args):
        value = self.griddef.resolution_tag
        if args[0] == "uppercase":
            value = value.upper()
        return value

    def _get_attr_source_mission_sensor(self, *args):
        mission_sensor = self.metadata.mission_sensor
        if args[0] == "uppercase":
            mission_sensor = mission_sensor.upper()
        elif args[0] == "lower":
            mission_sensor = mission_sensor.lower()
        return mission_sensor

    def _get_attr_source_mission_sensor_fn(self, *args):
        """ Same as source mission sensor, only a sanitized version for filenames """
        mission_sensor = self.metadata.mission_sensor
        for character in ["-"]:
            mission_sensor = mission_sensor.replace(character, "")
        if args[0] == "uppercase":
            mission_sensor = mission_sensor.upper()
        return mission_sensor

    def _get_attr_source_hemisphere(self, *args):
        if args[0] != "select":
            return self.hemisphere
        choices = {"north": args[1], "south": args[2]}
        return choices.get(self.hemisphere, "n/a")

    @staticmethod
    def _get_attr_uuid(*args):
        return str(uuid.uuid4())

    def _get_attr_startdt(self, dtfmt):
        return self.metadata.start_period.strftime(dtfmt)

    def _get_attr_stopdt(self, dtfmt):
        return self.info.stop_time.strftime(dtfmt)

    def _get_attr_geospatial_lat_min(self, *args):
        latitude = self.vars["latitude"]
        return self._get_attr_geospatial_str(np.nanmin(latitude))

    def _get_attr_geospatial_lat_max(self, *args):
        latitude = self.vars["latitude"]
        return self._get_attr_geospatial_str(np.nanmax(latitude))

    def _get_attr_geospatial_lon_min(self, *args):
        longitude = self.vars["longitude"]
        return self._get_attr_geospatial_str(np.nanmin(longitude))

    def _get_attr_geospatial_lon_max(self, *args):
        longitude = self.vars["longitude"]
        return self._get_attr_geospatial_str(np.nanmax(longitude))

    @staticmethod
    def _get_attr_geospatial_str(value):
        return "%.4f" % value

    def _get_attr_source_auxdata_sic(self, *args):
        return self.metadata.source_auxdata_sic

    def _get_attr_source_auxdata_snow(self, *args):
        return self.metadata.source_auxdata_snow

    def _get_attr_source_auxdata_sitype(self, *args):
        return self.metadata.source_auxdata_sitype

    def _get_attr_utcnow(self, *args):
        dt = self._creation_time
        return dt.strftime(args[0]) if re.match("%", args[0]) else dt.isoformat()

    def _get_attr_time_coverage_start(self, *args):
        dt = self.metadata.time_coverage_start
        return dt.strftime(args[0]) if re.match("%", args[0]) else dt.isoformat()

    def _get_attr_time_coverage_end(self, *args):
        dt = self.metadata.time_coverage_end
        return dt.strftime(args[0]) if re.match("%", args[0]) else dt.isoformat()

    def _get_attr_time_coverage_duration(self, *args):
        return self.metadata.time_coverage_duration

    def _get_attr_doi(self, *args):
        return self._doi

    def _get_attr_data_record_type(self, *args):
        if args[0] != "select":
            return self._data_record_type
        choices = {"cdr": args[1], "icdr": args[2]}
        return choices.get(self._data_record_type, "n/a")

    @staticmethod
    def _get_attr_pysiral_version(*args):
        return __version__

    def flipud(self):
        for parameter in self.parameters:
            setattr(self, parameter, np.flipud(getattr(self, parameter)))

    @property
    def metadata(self):
        return self._metadata

    @property
    def grid_xi_range(self):
        return np.arange(self.griddef.extent.numx)

    @property
    def grid_yj_range(self):
        return np.arange(self.griddef.extent.numy)

    @property
    def all_grid_indices(self):
        return itertools.product(self.grid_xi_range, self.grid_yj_range)

    @property
    def grid_indices(self):
        if self._non_empty_grid_indices is None:
            return itertools.product(self.grid_xi_range, self.grid_yj_range)
        else:
            return self._non_empty_grid_indices

    # @property
    # def parameter_list(self):
    #     # TODO: Only L2 parameter for now
    #     parameter_list = list(self._l2_parameter)
    #     parameter_list.extend(self._l3_parameter)
    #     parameter_list.append("lon")
    #     parameter_list.append("lat")
    #     return parameter_list

    @property
    def dimdict(self):
        time_dim = 0 if True in self._time_dim_is_unlimited else 1
        return OrderedDict([("time", time_dim),
                            ("yc", self.griddef.extent.numx),
                            ("xc", self.griddef.extent.numy)])

    @property
    def grid_shape(self):
        return self.griddef.extent.numx, self.griddef.extent.numy

    @property
    def griddef(self):
        return self._griddef

    @property
    def l3def(self):
        return self._l3def

    @property
    def hemisphere(self):
        return self.metadata.hemisphere

    @property
    def time_bounds(self):
        return [self.metadata.time_coverage_start,
                self.metadata.time_coverage_end]


class L3MetaData(object):
    """
    Container for L3S Metadata information
    (see property attribute_list for a list of attributes)
    """

    _attribute_list = [
        "mission_ids", "start_time", "stop_time", "grid_name", "period_label",
        "time_coverage_start", "time_coverage_end", "time_coverage_duration",
        "time_coverage_resolution", "pysiral_version", "projection_str", "grid_tag", "resolution_tag",
        "hemisphere", "mission_sensor", "source_auxdata_sic",
        "source_auxdata_sitype", "source_auxdata_snow", "source_timeliness"]

    def __init__(self):
        # Init all fields
        for field in self.attribute_list:
            setattr(self, field, None)

    def get_missions_from_stack(self, stack):
        """
        Get a list of missions that went into the stack
        (must be a list, since multi-mission grids are supported)
        """
        missions = np.unique(stack.mission)
        mission_sensor = [psrlcfg.platforms.get_sensor(mission.lower()) for mission in missions]

        try:
            self.set_attribute("mission_ids", ",".join(missions))
        except TypeError:
            self.set_attribute("mission_ids", "unkown")
        try:
            self.set_attribute("mission_sensor", ",".join(mission_sensor))
        except TypeError:
            self.set_attribute("mission_ids", "unkown")

        source_timeliness = np.unique(stack.timeliness)[0]
        self.set_attribute("source_timeliness", source_timeliness)

    def get_data_period_from_stack(self, stack):
        """ Get the first and last timestamp """
        self.set_attribute("start_time", np.amin(stack.start_time))
        self.set_attribute("stop_time", np.amax(stack.stop_time))
        # XXX: Only monthly periods are currently supported
        self.set_attribute("period_label", self.start_time.strftime("%B %Y"))

    def get_time_coverage_from_period(self, period):
        """ Get the start and end of requested data period """
        # TODO: replace with the period.get_netcdf_attributes() method
        self.set_attribute("time_coverage_start", period.tcs.dt)
        self.set_attribute("time_coverage_end", period.tce.dt)
        self.set_attribute("time_coverage_duration", period.duration.isoformat)

    def get_auxdata_infos(self, l2i_info):
        """
        Get information on auxiliary data sources from l2i global
        attributes
        TODO: This part is deprecated
        :param l2i_info:
        :return:
        """
        try:
            self.set_attribute("source_auxdata_sic", l2i_info.source_sic)
        except AttributeError:
            self.set_attribute("source_auxdata_sic", l2i_info.source_auxdata_sic)
        try:
            self.set_attribute("source_auxdata_sitype", l2i_info.source_sitype)
        except AttributeError:
            self.set_attribute("source_auxdata_sitype", l2i_info.source_auxdata_sitype)
        try:
            self.set_attribute("source_auxdata_snow", l2i_info.source_snow)
        except AttributeError:
            self.set_attribute("source_auxdata_snow", l2i_info.source_auxdata_snow)

    def get_projection_parameter(self, griddef):
        self.set_attribute("grid_tag", griddef.grid_tag)
        self.set_attribute("hemisphere", griddef.hemisphere)
        self.set_attribute("resolution_tag", griddef.resolution_tag)

    def __repr__(self):
        output = "pysiral.L3S Metadata Container:\n"
        for field in self._attribute_list:
            output += "%22s: %s" % (field, getattr(self, field))
            output += "\n"
        return output

    @property
    def attribute_list(self):
        return self._attribute_list

    @property
    def attdict(self):
        """ Return attributes as dictionary (e.g. for netCDF export) """
        return {field: getattr(self, field) for field in self.attribute_list}

    @property
    def mission(self):
        mission_ids = self.mission_ids.split(",")
        return "_".join(mission_ids)

    def set_attribute(self, tag, value):
        if tag not in self.attribute_list:
            raise ValueError("Unknown attribute: ", tag)
        setattr(self, tag, value)


class Level3OutputHandler(OutputHandlerBase):
    # Some fixed parameters for this class
    default_file_location = ["settings", "outputdef", "l3_default.yaml"]
    subfolder_tags = ["year"]
    applicable_data_level = 3

    def __init__(self, output_def="default", base_directory="l3proc_default",
                 overwrite_protection=True, period="default", doi=None,
                 data_record_type="none"):

        if output_def == "default":
            output_def = self.default_output_def_filename

        super(Level3OutputHandler, self).__init__(
            output_def, applicable_data_level=3, subfolder_tags=["year"],
            default_file_location=["settings", "outputdef", "l3_default.yaml"])

        self.error.caller_id = self.__class__.__name__
        logger.name = self.__class__.__name__

        self._period = period
        self._doi = doi
        self.overwrite_protection = overwrite_protection

        self._init_product_directory(base_directory)
        self._data_record_type = data_record_type

    def get_filename_from_data(self, l3):
        """ Return the filename for a defined level-3 data object
        based on tag filenaming in output definition file """

        # Get the filenaming definition (depending on period definition)
        filename_template = ""
        try:
            template_ids = self.output_def.filenaming.keys()
            period_id = self._period

            # Add a translation for the current dissonance between dateperiod
            # period id's and the pysiral convention
            # period_id_dict = dict(month="monthly", isoweek="weekly")
            # period_id = period_id_dict.get(period_id, period_id)
            # Fall back to default if no filenaming convention for given
            # data period
            if period_id not in template_ids:
                period_id = "default"
            filename_template = self.output_def.filenaming[period_id]
        except AttributeError:
            filename_template = self.output_def.filenaming
        except KeyError:
            msg = "Missing filenaming convention for period [%s] in [%s]"
            msg %= (str(self._period), self.output_def_filename)
            self.error.add_error("invalid-outputdef", msg)
            self.error.raise_on_error()

        return self.fill_template_string(filename_template, l3)

    def get_directory_from_data(self, l3, create=True):
        """ Return the output directory based on information provided
        in an l2 data object """
        directory = self._get_directory_from_dt(l3.metadata.start_time)
        if create:
            self._create_directory(directory)
        return directory

    def get_fullpath_from_data(self, l3):
        """ Return export path and filename based on information
        provided in the l2 data object """
        export_directory = self.get_directory_from_data(l3)
        export_filename = self.get_filename_from_data(l3)
        return Path(export_directory) / export_filename

    def get_global_attribute_dict(self, l3):
        attr_dict = OrderedDict()
        for attr_entry in self.output_def.global_attributes:
            attr_name, attr_template = zip(*attr_entry.items())
            attribute = self.fill_template_string(attr_template[0], l3)
            attr_dict[attr_name[0]] = attribute
        return attr_dict

    def _init_product_directory(self, base_directory_or_id):
        """ Initializes the product directory. If `base_directory` is already
        a directory, it is used as is. Else, it is treated as subfolder of
        the default pysiral product directory. Product level id and
        overwrite protection subfolders are added for both options"""
        # argument is directory
        if Path(base_directory_or_id).is_dir():
            basedir = Path(base_directory_or_id)
        # argument is id
        else:
            basedir = Path(self.pysiral_config.local_machine.product_repository) / base_directory_or_id
        # add product level subfolder
        # period_id = dict(month="monthly", isoweek="weekly")
        basedir = basedir / self.product_level_subfolder / self._period
        # optional (subfolder with current time)
        if self.overwrite_protection:
            basedir = basedir / self.now_directory
        # set the directory
        self._set_basedir(basedir)

    @property
    def flip_yc(self):
        flip_yc = self.output_def.grid_options.flip_yc
        if not isinstance(flip_yc, bool):
            flip_yc = False
        return flip_yc

    @property
    def doi(self):
        return self._doi

    @property
    def data_record_type(self):
        return self._data_record_type

    @property
    def time_dim_is_unlimited(self):

        # This property has been added. Older L3 output definitions may not have it,
        # -> Catch attribute error and return false if attribute does not exist
        if "time_dim_is_unlimited" not in self.output_def.grid_options:
            msg = "`grid_options.time_dim_is_unlimited` is missing in l3 settings file: %s (Using default: False)"
            logger.warning(msg % self.output_def_filename)
            time_dim_is_unlimited = False
        else:
            time_dim_is_unlimited = self.output_def.grid_options.time_dim_is_unlimited

        # Verification: Value must be bool
        if not isinstance(time_dim_is_unlimited, bool):
            msg = 'Invalid value type for `grid_options.time_dim_is_unlimited` in %s. ' + \
                  'Must be bool, value was %s. (Using default: False)'
            msg = msg % (self.output_def_filename, str(time_dim_is_unlimited))
            logger.error(msg)
            time_dim_is_unlimited = False

        return time_dim_is_unlimited


class Level3ProductDefinition(object):

    def __init__(self, l3_settings_file, grid, output, period):
        """ Container for the Level3Processor settings

        Arguments:
            l3_settings_file (str): Full filename to l3 settings file
            grid (pysiral.grid.GridDefinition): Output grid class
            output (Level-3 compliant output handler from pysiral.core.output)
        """
        self.error = ErrorStatus(caller_id=self.__class__.__name__)
        self._l3_settings_file = l3_settings_file
        self._output = output
        self._grid = grid
        self._period = period
        self._parse_l3_settings()

        # Report settings to log handler
        logger.info(f"Output grid id: {str(self._grid.grid_id)}")
        for output in self._output:
            msg = "L3 product directory (%s): %s"
            msg %= (str(output.id), str(output.basedir))
            logger.info(msg)

    def _parse_l3_settings(self):
        logger.info(f"Parsing settings: {str(self._l3_settings_file)}")
        try:
            self._l3 = get_yaml_as_dict(Path(self._l3_settings_file))
        except Exception as ex:
            self.error.add_error("l3settings-parser-error", str(ex))
            self.error.raise_on_error()

    def validate(self):
        pass

    @property
    def grid(self):
        return self._grid

    @property
    def outputs(self):
        return self._output

    @property
    def n_outputs(self):
        return len(self._output)

    @property
    def l3def(self):
        return dict(**self._l3)

    @property
    def period(self):
        return self._period

    @property
    def l3_masks(self):
        """ Return a sorted list of the masks applied to level 3 data """
        try:
            mask_names = sorted(self.l3def["l3_masks"].keys())
            return [self.l3def["l3_masks"][name] for name in mask_names]
        except AttributeError:
            return []

    @property
    def l3_external_masks(self):
        try:
            extmask_names = self.l3def["external_masks"]
        except AttributeError:
            extmask_names = []
        return extmask_names

    @property
    def l3_post_processors(self):
        try:
            names = sorted(self.l3def["l3_post_processing"].keys(
                recursive=False, branch_mode="only"))
        except AttributeError:
            return []
        options = [self.l3def["l3_post_processing"][n]["options"] for n in names]
        return zip(names, options)

    @property
    def l2_parameter(self):
        """ Extract a list of paramter names to be extracted from
        l2i product files """
        return self.l3def["l2_parameter"]

    @property
    def l3_parameter(self):
        """ Extract a list of paramter names to be computed by the
        Level-3 processor """
        l3_parameter = sorted(self.l3def["l3_parameter"].keys())
        return [self.l3def["l3_parameter"][n] for n in l3_parameter]



