# -*- coding: utf-8 -*-
"""
This module contains the class `_PysiralPackageConfiguration` which is used to read and manage
the configuration files of the pysiral _package. It provides methods to access processor settings, output definitions,
and auxiliary data set definitions based on the configuration files located in the _package or user home directory.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = ["PysiralPackageConfiguration", "get_cls", "import_submodules", "set_psrl_cpu_count"]

import multiprocessing

import shutil
import socket
import sys
from datetime import timezone

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc

from pathlib import Path
from typing import Iterable, Dict

import yaml
from dateperiods import DatePeriod

import pysiral
from pysiral._package.auxdata_def import _AuxdataCatalogue
from pysiral._package.mission_def import _MissionDefinitionCatalogue
from pysiral._package.helper import get_cls, import_submodules, set_psrl_cpu_count

PACKAGE_ROOT_DIR = Path(pysiral.__file__).parent.resolve()


class PysiralPackageConfiguration(object):
    """
    Container for the content of the pysiral definition files
    (in pysiral/configuration) and the local machine definition file
    (local_machine_definition.yaml)
    """

    # --- Global variables of the pysiral _package configuration ---

    # Filenames of definitions files
    _DEFINITION_FILES = {
        "platforms": "mission_def.yaml",
        "auxdata": "auxdata_def.yaml",
    }

    # name of the file containing the data path on the local machine
    _LOCAL_MACHINE_DEF_FILE = "local_machine_def.yaml"

    # valid settings types, processor levels and data level ids.
    # NOTE: These tie into the naming and content of definition files
    VALID_SETTING_TYPES = ["proc", "output", "grid"]
    VALID_PROCESSOR_LEVELS = ["l1", "l2", "l2p", "l3"]
    VALID_DATA_LEVEL_IDS = ["l1", "l2", "l2i", "l2p", "l3", None]
    VALID_CONFIG_TARGETS = ["PACKAGE", "USER_HOME"]

    # Multiprocessing properties
    # Allow to _package-wide specification of number of CPU's. Default value
    # is the CPU count from the python multiprocessing _package.
    #
    # NOTE: This is intended when `multiprocessing.cpu_count()` is unreliable, e.g.
    #       when using slurm workload managers, or when the number of CPU's should be
    #       limited due to other concerns.
    CPU_COUNT = multiprocessing.cpu_count()

    def __init__(self):
        """
        Collect _package configuration data from the various definition files and provide an interface
        to pysiral processor, output and grid definition files.
        This class is intended to be only called inside the init module of pysiral and to store the
        pysiral _package configuration in the global variable `psrlcfg`
        """

        # --- Establish the path information ---
        # This step gets the default path (user home, set path for the resources)
        # NOTE: The current path to the active pysiral _package is already set in the global
        #       variable `pysiral.PACKAGE_ROOT_DIR`, since this is required for reading the
        #       version file
        self._path = self._get_pysiral_path_information()

        # --- Check the dedicated pysiral config path ---
        # As per default, the pysiral config path is set in the user home directory.
        self._check_pysiral_config_path()

        # --- Read the configuration files ---
        self.local_machine = self._read_local_machine_file()
        self.platforms = self._read_platforms()
        self.auxdata = self._read_auxdata_def()

    def _get_pysiral_path_information(self) -> Dict[str, str]:
        """
        Get the different path information for pysiral. This method will add the following
        attributes to self.path:
            1. package_root_path: The root directory of this _package
            2. package_config_path: The directory of the pysiral config items in this _package
            3. userhome_config_dir: The intended configuration directory in the user home
            4. config_target: The value given in the `PYSIRAL-CFG-LOC` file
        :return: None
        """

        # Store the root dir of this pysiral _package
        path = {"package_root_path": PACKAGE_ROOT_DIR}

        # Get the config directory of the _package
        # NOTE: This approach should work for a local script location or an installed _package
        path["package_config_path"] = path["package_root_path"] / "resources" / "pysiral-cfg"

        # Get an indication of the location for the pysiral configuration path
        # NOTE: In its default version, the text file `PYSIRAL-CFG-LOC` does only contain the
        #       string `USER_HOME`. In this case, pysiral will expect the a .pysiral-cfg sub-folder
        #       in the user home. The only other valid option is an absolute path to a specific
        #       directory with the same content as .pysiral-cfg. This was introduced to enable
        #       fully encapsulated pysiral installation in virtual environments

        # Get the home directory of the current user
        path["userhome_config_path"] = Path.home() / ".pysiral-cfg"

        # Read pysiral config location indicator file
        cfg_loc_file = PACKAGE_ROOT_DIR / "PYSIRAL-CFG-LOC"
        try:
            with open(str(cfg_loc_file)) as fh:
                path["config_target"] = fh.read().strip()
        except IOError:
            sys.exit(f"Cannot find PYSIRAL-CFG-LOC file in _package (expected: {cfg_loc_file})")

        return path

    def _check_pysiral_config_path(self):
        """
        This class ensures that the pysiral configuration files are in the chosen
        configuration directory
        :return:
        """

        # Make alias of
        config_path = Path(self.config_path)
        package_config_path = Path(self.path["package_config_path"])

        # Check if current config dir is _package config dir
        # if yes -> nothing to do (files are either there or aren't)
        if config_path == package_config_path:
            return

        # current config dir is not _package dir and does not exist
        # -> must be populated with content from the _package config dir
        if not config_path.is_dir():
            print(f"Creating pysiral config directory: {config_path}")
            shutil.copytree(str(self.path.package_config_path), str(config_path), dirs_exist_ok=True)
            print("Init local machine def")
            template_filename = package_config_path / "templates" / "local_machine_def.yaml"
            target_filename = config_path / "local_machine_def.yaml"
            shutil.copy(str(template_filename), str(target_filename))

    def _read_platforms(self) -> _MissionDefinitionCatalogue:
        """
        Read the three main configuration files for
            1. supported platforms
            2. supported auxiliary datasets
            3. path on local machine
        and create the necessary catalogues
        :return:
        """

        # --- Get information of supported platforms ---
        # The general information for supported radar altimeter missions (mission_def.yaml)
        # provides general metadata for each altimeter missions that can be used to sanity checks
        # and queries for sensor names etc.
        #
        # NOTE: This is just general information on altimeter platform and not to be confused with
        #       settings for actual primary data files. These are located in each l1p processor
        #       definition file.
        self.mission_def_filepath = self.config_path / Path(self._DEFINITION_FILES["platforms"])
        if not self.mission_def_filepath.is_file():
            error_msg = "Cannot load pysiral _package files: \n %s" % self.mission_def_filepath
            print(error_msg)
            sys.exit(1)
        return _MissionDefinitionCatalogue(self.mission_def_filepath)

    def _read_auxdata_def(self) -> _AuxdataCatalogue:

        # --- Get information on supported auxiliary data sets ---
        # The auxdata_def.yaml config file contains the central definition of the properties
        # of supported auxiliary data sets. Each auxiliary data set is uniquely defined by
        # the type of auxiliary data set and a name id.
        # The central definition allows accessing auxiliary data by its id in processor definition files
        self.auxdata_def_filepath = self.config_path / self._DEFINITION_FILES["auxdata"]
        if not self.auxdata_def_filepath.is_file():
            error_msg = "Cannot load pysiral _package files: \n %s" % self.auxdata_def_filepath
            print(error_msg)
            sys.exit(1)
        return _AuxdataCatalogue(self.auxdata_def_filepath)

        # read the local machine definition file

    @staticmethod
    def get_yaml_config(filename) -> Dict:
        """
        Read a yaml file and return it content as an attribute-enabled dictionary
        :param filename: path to the yaml file
        :return: attrdict.AttrDict
        """
        with open(str(filename)) as fileobj:
            settings = yaml.safe_load(fileobj)
        return settings

    def get_setting_ids(self, settings_type, data_level=None):
        lookup_directory = self.get_local_setting_path(settings_type, data_level)
        ids, files = self.get_yaml_setting_filelist(lookup_directory)
        return ids

    def get_platform_period(self, platform_id):
        """
        Get a period definition for a given platform ID
        :param platform_id:
        :return: dateperiods.DatePeriod
        """
        tcs, tce = self.platforms.get_time_coverage(platform_id)
        return DatePeriod(tcs, tce)

    def get_processor_definition_ids(self, processor_level):
        """
        Returns a list of available processor definitions ids for a given processor
        level (see self.VALID_PROCESSOR_LEVELS)
        :param processor_level:
        :return:
        """
        lookup_directory = self.get_local_setting_path("proc", processor_level)
        return self.get_yaml_setting_filelist(lookup_directory, return_value="ids")

    def get_settings_files(self, settings_type: str, data_level: str) -> Iterable[Path]:
        """
        Returns all processor settings or output definitions files for a given data level.
        :param settings_type:
        :param data_level:
        :return:
        """

        if settings_type not in self.VALID_SETTING_TYPES:
            return []

        if data_level not in self.VALID_DATA_LEVEL_IDS:
            return []

        # Get all settings files in settings/{data_level} and its
        # subdirectories
        lookup_directory = self.get_local_setting_path(settings_type, data_level)
        _, files = self.get_yaml_setting_filelist(lookup_directory)

        # Test if ids are unique and return error for the moment
        return files

    def get_settings_file(self, settings_type, data_level, setting_id_or_filename):
        """ Returns a processor settings file for a given data level.
        (data level: l2 or l3). The second argument can either be a
        direct filename (which validity will be checked) or an id, for
        which the corresponding file (id.yaml) will be looked up in
        the default directory """

        if settings_type not in self.VALID_SETTING_TYPES:
            return None

        if data_level not in self.VALID_DATA_LEVEL_IDS:
            return None

        # Check if filename
        if Path(setting_id_or_filename).is_file():
            return setting_id_or_filename

        # Get all settings files in settings/{data_level} and its
        # subdirectories
        lookup_directory = self.get_local_setting_path(settings_type, data_level)
        ids, files = self.get_yaml_setting_filelist(lookup_directory)

        # Test if ids are unique and return error for the moment
        if len(set(ids)) != len(ids):
            msg = f"Non-unique {settings_type}-{str(data_level)} setting filename"
            print(f"ambiguous-setting-files: {msg}")
            sys.exit(1)

        # Find filename to setting_id
        try:
            index = ids.index(setting_id_or_filename)
            return Path(files[index])
        except (IOError, ValueError):
            return None

    @staticmethod
    def get_yaml_setting_filelist(directory, return_value="both"):
        """ Retrieve all yaml files from a given directory (including
        subdirectories). Directories named "obsolete" are ignored if
        ignore_obsolete=True (default) """
        setting_ids = []
        setting_files = []
        for filepath in directory.rglob("*.yaml"):
            setting_ids.append(filepath.name.replace(".yaml", ""))
            setting_files.append(filepath)
        if return_value == "both":
            return setting_ids, setting_files
        elif return_value == "ids":
            return setting_ids
        elif return_value == "files":
            return setting_files
        else:
            raise ValueError(f"Unknown return value {str(return_value)} [`both`, `ids`, `files`]")

    def get_local_setting_path(self, settings_type, data_level=None):
        """
        Return the absolute path on the local productions system to the configuration file. The
        returned path depends on the fixed structure below the `resources` directory in the pysiral
        _package and the choice in the config file "PYSIRAL-CFG-LOC"
        :param settings_type:
        :param data_level:
        :return:
        """
        if settings_type in self.VALID_SETTING_TYPES and data_level in self.VALID_DATA_LEVEL_IDS:
            args = [settings_type]
            if data_level is not None:
                args.append(data_level)
            return Path(self.config_path) / Path(*args)
        else:
            return None

    def reload(self):
        """
        Method to trigger reading the configuration files again, e.g. after changing the config target
        :return:
        """
        self._read_config_files()
        self._check_pysiral_config_path()

    def set_config_target(self, config_target, permanent=False):
        """
        Set the configuration target
        :param config_target:
        :param permanent:
        :return:
        """

        # Input validation
        if config_target in self.VALID_CONFIG_TARGETS or Path(config_target).is_dir():
            self._path["config_target"] = config_target
        else:
            msg = "Invalid config_target: {} must be {} or valid path"
            msg = msg.format(str(config_target), ", ".join(self.VALID_CONFIG_TARGETS))
            raise ValueError(msg)

        if permanent:
            raise NotImplementedError()

    def _read_local_machine_file(self) -> Dict:
        """
        :return:
        """
        filename = self.local_machine_def_filepath
        try:
            local_machine_def = self.get_yaml_config(filename)
        except IOError:
            msg = f"local_machine_def.yaml not found (expected: {filename})"
            # print(f"local-machine-def-missing: {msg}")
            local_machine_def = None
        return local_machine_def

    @property
    def platform_ids(self):
        return self.platforms.ids

    @property
    def path(self):
        return self._path

    @property
    def userhome_config_path(self):
        return Path(self.path["userhome_config_path"])

    @property
    def package_config_path(self):
        return Path(self.path["package_config_path"])

    @property
    def package_path(self):
        return Path(PACKAGE_ROOT_DIR)

    @property
    def current_config_target(self):
        return str(self._path["config_target"])

    @property
    def config_target(self):
        return str(self._path["config_target"])

    @property
    def config_path(self):
        """
        nstruct the target config path based on the value in `PYSIRAL-CFG-LOC`
        :return:
        """
        # Case 1 (default): pysiral config path is in user home
        if self._path["config_target"] == "USER_HOME":
            return Path(self._path["userhome_config_path"])

        # Case 2: pysiral config path is the _package itself
        elif self._path["config_target"] == "PACKAGE":
            return Path(self._path["package_config_path"])

        # Case 3: _package specific config path
        else:
            # This should be an existing path, but in the case it is not, it is created
            return Path(self._path["config_target"])

    @property
    def local_machine_def_filepath(self):
        if self.current_config_target != "PACKAGE":
            return self.config_path / self._LOCAL_MACHINE_DEF_FILE
        # TODO: Disable warnings that are run on simple import (as long as everything runs)
        # msg = "Current config path is `PACKAGE`, lookup directory for local_machine_def.yaml changed to `USERHOME`"
        # logger.warning(msg)
        return self.userhome_config_path / self._LOCAL_MACHINE_DEF_FILE

    @property
    def processor_levels(self):
        return list(self.VALID_PROCESSOR_LEVELS)

    @property
    def hostname(self):
        return socket.gethostname()

    @property
    def version(self):
        return str(pysiral.__version__)
