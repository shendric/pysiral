from datetime import datetime
from netCDF4 import Dataset, date2num
from pathlib import Path
from typing import Optional

import cftime
import numpy as np
from loguru import logger

from dateperiods import DatePeriod
from pysiral import PSRLCFG
from pysiral.core.legacy_classes import DefaultLoggingClass, AttrDict, ErrorStatus
from pysiral.core.output import NCDateNumDef
from pysiral.l1 import Level1bData


class Level1POutputHandler(DefaultLoggingClass):
    """
    The output handler for l1p product files
    NOTE: This is not a subclass of OutputHandlerbase due to the special nature of pysiral l1p products
    TODO: Refactoring needed. Path needs to be function and not a property that must be set before export.
    """

    def __init__(self, cfg: AttrDict) -> None:
        cls_name = self.__class__.__name__
        super(Level1POutputHandler, self).__init__(cls_name)
        self.error = ErrorStatus(caller_id=cls_name)
        self.cfg = cfg

        self.pysiral_cfg = PSRLCFG
        self._file_log = []

    @staticmethod
    def remove_old_if_applicable(perid: DatePeriod) -> None:
        logger.warning("Not implemented: self.remove_old_if_applicable")
        return

    def export_to_netcdf(self, l1: "Level1bData") -> None:
        """
        Workflow to export a Level-1 object to l1p netCDF product. The workflow includes the generation of the
        output path (if applicable).

        :param l1: The Level-1 object to be exported

        :return: None
        """

        # Check file length before exporting
        # NOTE: This has been added
        minimum_file_record_size = self.cfg.get("minimum_file_record_size", 0)
        if l1.n_records < minimum_file_record_size:
            logger.warning(f"- Orbit segment too short: {l1.n_records} < {minimum_file_record_size} (min), "
                           "skipping export to file")
            return

        # Get filename and path
        output_filepath = self.get_output_filepath(l1)

        # Check if path exists
        Path(output_filepath).parent.mkdir(exist_ok=True, parents=True)

        # Export the data object
        ncfile = L1bDataNC()
        ncfile.l1b = l1
        ncfile.output_folder = output_filepath.parent
        ncfile.filename = output_filepath.name
        ncfile.export()

        self._file_log.append(output_filepath)

    def get_output_filepath(self, l1: "Level1bData") -> Path:
        """
        Sets the class properties required for the file export

        :param l1: The Level-1 object

        :return: None
        """

        local_machine_def_tag = self.cfg.get("local_machine_def_tag", None)
        if local_machine_def_tag is None:
            msg = "Missing mandatory option %s in l1p processor definition file -> aborting"
            msg %= "root.output_handler.options.local_machine_def_tag"
            msg = msg + "\nOptions: \n" + self.cfg.makeReport()
            self.error.add_error("missing-option", msg)
            self.error.raise_on_error()

        # TODO: Move to config file
        filename_template = "pysiral-l1p-{platform}-{source}-{timeliness}-{hemisphere}-{tcs}-{tce}-{file_version}.nc"
        time_fmt = "%Y%m%dT%H%M%S"
        values = {"platform": l1.info.mission,
                  "source": self.cfg.version.source_file_tag,
                  "timeliness": l1.info.timeliness,
                  "hemisphere": l1.info.hemisphere,
                  "tcs": l1.time_orbit.timestamp[0].strftime(time_fmt),
                  "tce": l1.time_orbit.timestamp[-1].strftime(time_fmt),
                  "file_version": self.cfg.version.version_file_tag}
        filename = filename_template.format(**values)

        local_repository = PSRLCFG.local_machine.l1b_repository
        export_folder = Path(local_repository[l1.info.mission][local_machine_def_tag]["l1p"])
        yyyy = "%04g" % l1.time_orbit.timestamp[0].year
        mm = "%02g" % l1.time_orbit.timestamp[0].month
        return (
            export_folder /
            self.cfg.version.get("source_file_tag", "unknown_source") /
            self.cfg.version.get("version_file_tag", "unknown_version") /
            l1.info.hemisphere /
            yyyy /
            mm /
            filename
        )

    @property
    def last_written_file(self) -> Optional[Path]:
        try:
            return self._file_log[-1]
        except IndexError:
            return None


class L1bDataNC(DefaultLoggingClass):
    """
    Class to export a L1bdata object into a netcdf file
    NOTE: This class is different from the other netCDF output classes as
          it uses data groups.
    """

    def __init__(self):

        super(L1bDataNC, self).__init__(self.__class__.__name__)

        self.datagroups = ["waveform", "surface_type", "time_orbit", "classifier", "correction"]
        self.filename = None
        self.time_def = NCDateNumDef()
        self.zlib = True
        self._rootgrp = None
        self._options = None
        self._proc_settings = None
        self.verbose = False

        self.output_folder = None
        self.l1b = None
        # TODO: Remove parameter attributes altogether with l1 setting files
        self.parameter_attributes = []

    def export(self):
        self._validate()
        self._open_file()
        # Save the l1b info data group as global attributes
        attdict = self.l1b.info.attdict
        self._create_root_group(attdict)
        self._populate_data_groups()
        self._write_to_file()

    def _validate(self):
        if self.filename is None:
            self._create_filename()
        self.path = Path(self.output_folder) / self.filename

    def _create_filename(self):
        self.filename = Path(f'{self.l1b.filename.stem}.nc')

    def _set_global_attributes(self, attdict, prefix=""):
        """ Save l1b.info dictionary as global attributes """
        for key in sorted(attdict.keys()):
            self._rootgrp.setncattr(prefix+key, attdict[key])

    def _create_root_group(self, attdict, **global_attr_keyw):
        """
        Create the root group and add l1b metadata as global attributes
        :param attdict: A dictionary containing the global attributes
        :param global_attr_keyw:
        :return:
        """
        self._convert_datetime_attributes(attdict)
        self._convert_bool_attributes(attdict)
        self._convert_nonetype_attributes(attdict)
        self._set_global_attributes(attdict, **global_attr_keyw)

    def _populate_data_groups(self):

        self._missing_parameters = []

        for datagroup in self.datagroups:

            if self.verbose:
                print(datagroup.upper())

            # Create the data group
            dgroup = self._rootgrp.createGroup(datagroup)
            content = getattr(self.l1b, datagroup)

            # Create the dimensions
            # (must be available as OrderedDict in data group container
            dims = content.dimdict.keys()
            for key in dims:
                dgroup.createDimension(key, content.dimdict[key])

            # Now add variables for each parameter in data group
            for parameter in content.parameter_list:

                data = getattr(content, parameter)

                # Convert datetime objects to number
                if isinstance(data[0], (datetime, cftime.datetime, cftime.real_datetime)):
                    data = date2num(data, self.time_def.units, self.time_def.calendar)

                # Convert bool objects to integer
                if data.dtype.str == "|b1":
                    data = np.int8(data)
                dimensions = tuple(list(dims)[:len(data.shape)])
                if self.verbose:
                    print(f" {parameter}", dimensions, data.dtype.str, data.shape)

                var = dgroup.createVariable(parameter, data.dtype.str, dimensions, zlib=self.zlib)
                var[:] = data

                # Add Parameter Attributes
                attribute_dict = self._get_variable_attr_dict(parameter)
                for key in attribute_dict.keys():
                    setattr(var, key, attribute_dict[key])

    def _convert_datetime_attributes(self, attdict):
        """
        Change any datetime-like object to a netCDF compatible numerical representation
        :param attdict: dictionary-like
        :return:
        """

        for key in attdict.keys():
            content = attdict[key]
            if isinstance(content, (datetime, cftime.datetime, cftime.real_datetime)):
                attdict[key] = date2num(content, self.time_def.units, self.time_def.calendar)

    @staticmethod
    def _convert_bool_attributes(attdict):
        """
        Replace l1b info parameters of type bool ['b1'] by a integer representation
        to match requirements for netCDF attribute data type rules
        :param attdict: dictionary-like
        :return:
        """
        for key in attdict.keys():
            content = attdict[key]
            if type(content) is bool:
                attdict[key] = int(content)

    @staticmethod
    def _convert_nonetype_attributes(attdict):
        """
        Change empty (None) values in an attrdict to an empty string
        :param attdict: dictionary like
        :return: None, will be changed in place
        """
        for key in attdict.keys():
            content = attdict[key]
            if content is None:
                attdict[key] = ""

    def _get_variable_attr_dict(self, parameter):
        """ Retrieve the parameter attributes """
        default_attrs = {"long_name": parameter}
        if parameter not in self.parameter_attributes:
            # self._missing_parameters.append(parameter)
            return default_attrs
        else:
            return dict(self.parameter_attributes[parameter])

    def _open_file(self):
        self._rootgrp = Dataset(self.path, "w")

    def _write_to_file(self):
        self._rootgrp.close()
