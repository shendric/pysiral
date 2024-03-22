# -*- coding: utf-8 -*-

"""
Internal module with classes for Input/Output operations of
the Level-1 pre-processor
"""

from pathlib import Path
from typing import Union

from attrdict import AttrDict
from loguru import logger

from pysiral import psrlcfg
from pysiral.core.output import L1bDataNC
from pysiral.l1data import Level1bData


class SourceFileDiscovery(object):
    """
    This class collects all Level-1 input handler classes
    (A catalog of classes is created when sub-classing this class)

    Usage:
        Level1InputHandler.get_class(class_name, **kwargs)
    """

    def __init_subclass__(cls, **kwargs):
        # if "search" not in cls.__dict__:
        #     raise NotImplementedError(f"class {cls.__name__} does not implement method `search`")
        psrlcfg.registered_classes.source_file_discovery[cls.__name__] = cls

    @classmethod
    def get_cls(cls, class_name: str, **kwargs):
        return psrlcfg.registered_classes.source_file_discovery[class_name](kwargs)


class SourceDataLoader(object):
    """
    This class collects all Level-1 input handler classes
    (A catalog of classes is created when sub-classing this class)

    Usage:
        Level1InputHandler.get_class(class_name, **kwargs)
    """

    def __init_subclass__(cls, **kwargs):
        # if "search" not in cls.__dict__:
        #     raise NotImplementedError(f"class {cls.__name__} does not implement method `search`")
        psrlcfg.registered_classes.source_data_input[cls.__name__] = cls

    @classmethod
    def get_cls(cls, class_name: str, **kwargs):
        return psrlcfg.registered_classes.source_data_input[class_name](kwargs)


# TODO: To be refactord to "Input Adapter base"


class Level1POutputHandler(object):
    """
    The output handler for l1p product files
    NOTE: This is not a subclass of OutputHandlerbase due to the special nature of pysiral l1p products
    """

    def __init__(self, cfg: AttrDict) -> None:
        self.cfg = cfg
        self._path = None
        self._filename = None

    @staticmethod
    def remove_old_if_applicable() -> None:
        logger.warning("Not implemented: self.remove_old_if_applicable")
        return

    def export_to_netcdf(self, l1: "Level1bData") -> None:
        """
        Workflow to export a Level-1 object to l1p netCDF product. The workflow includes the generation of the
        output path (if applicable).

        :param l1: The Level-1 object to be exported

        :return: None
        """

        # Get filename and path
        self.set_output_filepath(l1)

        # Check if path exists
        Path(self.path).mkdir(exist_ok=True, parents=True)

        # Export the data object
        ncfile = L1bDataNC()
        ncfile.l1b = l1
        ncfile.output_folder = self.path
        ncfile.filename = self.filename
        ncfile.export()

    def set_output_filepath(self, l1: "Level1bData") -> None:
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
            raise KeyError(msg)

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
        self._filename = filename_template.format(**values)

        local_repository = psrlcfg.local_machine.l1b_repository
        export_folder = Path(local_repository[l1.info.mission][local_machine_def_tag]["l1p"])
        yyyy = "%04g" % l1.time_orbit.timestamp[0].year
        mm = "%02g" % l1.time_orbit.timestamp[0].month
        self._path = export_folder / self.cfg.version["version_file_tag"] / l1.info.hemisphere / yyyy / mm

    @property
    def path(self) -> Path:
        return Path(self._path)

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def last_written_file(self) -> Path:
        return self.path / self.filename