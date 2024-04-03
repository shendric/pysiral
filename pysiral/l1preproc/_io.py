# -*- coding: utf-8 -*-

"""
Internal module with classes for Input/Output operations of
the Level-1 pre-processor
"""

from pathlib import Path
from typing import List
import inspect

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
        SourceDataLoader.get_class(class_name, **kwargs)
    """

    def __init_subclass__(cls, supported_source_datasets: List[str] = None):
        """
        Registers a class as source data discovery class for specific source data sets
        in the pysiral package configuration. This processes includes a basic test that
        the class is compliant with requirements for a SourceFileDiscovery class, namely
        a test if certain methods with correct return types have been implemented.

        :param supported_source_datasets: A list of source data set id's supported by the class

        :raises NotImplementedError: Subclass does not implement all required
            methods.

        :return: None
        """

        logger.debug(f"SourceFileDiscovery: Register {cls} with {supported_source_datasets=}")

        # Any subclass must have a method that returns a
        check_class_compliance(
            cls,
            "query_period",
            "typing.List[pathlib.Path]"
        )

        # Input checks passed -> register class for source data sets
        # Note: Allow to overwrite already registered classes, but
        #       warn of overwrite.
        for supported_dataset in supported_source_datasets:
            existing_cls = psrlcfg.registered_classes.source_data_discovery.get(supported_dataset)
            if existing_cls is cls:
                return  # Class just being imported again
            if existing_cls is not None:
                logger.warning(
                    f"Source data discovery class {existing_cls} will be overwritten by {cls} "
                    f"for dataset id={supported_dataset}"
                )
            psrlcfg.registered_classes.source_data_discovery[supported_dataset] = cls

    @classmethod
    def get_cls(cls, source_dataset_id: str, **kwargs):
        """
        Get the initialized source data discovery class.

        :param source_dataset_id: A pysiral known source dataset identifier
        :param kwargs: keyword arguments for the source data discovery class

        :raises KeyError: Invalid source_dataset_id.

        :return: Initialized source data discovery class
        """

        lookup_directory = psrlcfg.local_path.get_source_directory(source_dataset_id)
        try:
            target_cls = psrlcfg.registered_classes.source_data_discovery[source_dataset_id]
            return target_cls(lookup_directory, **kwargs)
        except KeyError as ke:
            msg = (
                f"Could not find SourceFileDiscovery implementation for {source_dataset_id=} "
                f"[Available: {list(psrlcfg.registered_classes.source_data_discovery.keys())}]"
            )
            raise KeyError(msg) from ke


class SourceDataLoader(object):
    """
    This class collects all Level-1 input handler classes
    (A catalog of classes is created when sub-classing this class)

    Usage:
        SourceDataLoader.get_class(class_name, **kwargs)
    """

    def __init_subclass__(cls, supported_source_datasets: List[str] = None) -> None:
        """
        Registers a class as source data discovery class for specific source data sets
        in the pysiral package configuration. This processes includes a basic test that
        the class is compliant with requirements for a SourceFileDiscovery class, namely
        a test if certain methods with correct return types have been implemented.

        :param supported_source_datasets: A list of source data set id's supported by the class

        :raises NotImplementedError: Subclass does not implement all required
            methods.

        :return: None
        """

        logger.debug(f"SourceDataLoader: Register {cls} with {supported_source_datasets=}")

        # Input class validation
        check_class_compliance(
            cls,
            "get_l1",
            "typing.Optional[pysiral.l1data.Level1bData]"
        )

        # Input checks passed -> register class for source data sets
        # Note: Allow to overwrite already registered classes, but
        #       warn of overwrite.
        for supported_dataset in supported_source_datasets:
            existing_cls = psrlcfg.registered_classes.source_data_loader.get(supported_dataset)
            if existing_cls is not None:
                logger.warning(
                    f"Source data discovery class {existing_cls} will be overwritten by {cls} "
                    "for dataset id={supported_dataset}"
                )
            psrlcfg.registered_classes.source_data_loader[supported_dataset] = cls

    @classmethod
    def get_cls(cls, source_dataset_id: str, **kwargs):
        """
        Fetches and initialized the source data loader class for a given
        source dataset identifier.

        :param source_dataset_id: Source dataset identifier
        :param kwargs: Keyword arguments specific to source data loader class

        :raises KeyError: No registered source data loader class for given source dataset id

        :raises TypeError: Incorrect keyword arguments to source data loader class

        :return: Initialized source data loader class
        """
        try:
            target_cls = psrlcfg.registered_classes.source_data_loader[source_dataset_id]
        except KeyError as ke:
            registered_classes = list(psrlcfg.registered_classes.source_data_loader.keys())
            msg = f"Cannot find source data loader class for {source_dataset_id=} {registered_classes}"
            raise KeyError(msg) from ke

        try:
            return target_cls(**kwargs)
        except TypeError as te:
            msg = f"Incorrect input {kwargs=} to {target_cls=}"
            raise TypeError(msg) from te


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


def check_class_compliance(
        cls,
        required_method_name: str,
        required_return_annotation
) -> None:
    """
    Check that class cls has a required method, which has the correct return data type.
    The return data type is only inferred from the rethr type annotation.

    :param cls:
    :param required_method_name:
    :param required_return_annotation:

    :raises NotImplementedError: Test fails

    :return: None
    """

    # Input class validation
    if required_method_name not in cls.__dict__:
        raise NotImplementedError(f"class {cls.__name__} does not implement method`: {required_method_name}")
    method_signature = inspect.signature(getattr(cls, required_method_name))
    if str(method_signature.return_annotation) != required_return_annotation:
        raise NotImplementedError(
            f"class {cls.__name__}.{required_method_name} "
            f"does not return {required_return_annotation} "
            f"but {str(method_signature.return_annotation)}"
        )
