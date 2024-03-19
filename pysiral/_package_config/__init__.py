# -*- coding: utf-8 -*-

"""

"""

import multiprocessing
import socket
import platform
import psutil
import sys
from pathlib import Path
from typing import Union, Dict, List, Literal
from pydantic import BaseModel, DirectoryPath, ConfigDict, PositiveInt, PositiveFloat
from ruamel.yaml import YAML

from ._auxdata import AuxiliaryDataConfig
from ._local_machine import LocalMachineConfig
from ._mission import PlatformConfig
from ._output import OutputDefCatalog
from ._proc import ProcDefCatalog


# Filenames of definitions files
_DEFINITION_FILES = {
    "auxdata": "auxdata_def.yaml",
}

# name of the file containing the data path on the local machine
_LOCAL_MACHINE_DEF_FILE = "local_machine_def.yaml"

# valid settings types, processor levels and data level ids.
# NOTE: These tie into the naming and content of definition files
VALID_SETTING_TYPES = ["proc", "output", "grid"]
VALID_PROCESSOR_LEVELS = ["l1", "l2", "l3"]
VALID_DATA_LEVEL_IDS = ["l1", "l2", "l2i", "l2p", "l3", None]
VALID_CONFIG_TARGETS = ["PACKAGE", "USER_HOME"]


__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


class PackageConfig(BaseModel):
    """
    Includes the path information to the package configuration (.pysiral-cfg/*)
    """
    package_root: DirectoryPath
    package_config: DirectoryPath
    user_home: DirectoryPath
    config_target: Union[Literal["PACKAGE", "USER_HOME"], DirectoryPath]

    @property
    def local_machine_def_filepath(self) -> Union[Path, None]:
        """
        The local machine def config file is something that requires
        manual interaction after package installation. It is therefore
        allowed to be not defined, though nothing will work without the
        file in production

        :return: Path to `local_machine_def.yaml` or None if it hasn't been created yet
        """
        if self.config_target != "PACKAGE":
            filepath = self.config_path / _LOCAL_MACHINE_DEF_FILE
        else:
            filepath = self.user_home / ".pysiral-cfg" / _LOCAL_MACHINE_DEF_FILE
        return filepath if filepath.is_file() else None

    @property
    def config_path(self) -> Path:
        """
        nstruct the target config path based on the value in `PYSIRAL-CFG-LOC`
        :return:
        """
        # Case 1 (default): pysiral config path is in user home
        if self.config_target == "USER_HOME":
            return self.user_home / ".pysiral-cfg"

        # Case 2: pysiral config path is the package itself
        elif self.config_target == "PACKAGE":
            return self.package_config

        # Case 3: package specific config path
        else:
            # This should be an existing path, but in the case it is not, it is created
            return self.config_target


class SystemConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    platform: str
    platform_release: str
    platform_version: str
    architecture: str
    hostname: str
    processor: str
    memory_gb: PositiveFloat
    mp_cpu_count: PositiveInt
    python_version: str


class _PysiralPackageConfiguration(object):
    """
    Container for the content of the pysiral definition files
    (in pysiral/configuration) and the local machine definition file
    (local_machine_definition.yaml)

    Main Properties
    - system (system properties)
    - package (pysiral package information, e.g. config path etc.)
    - platforms (information on supported platforms)
    - local_path (Path to data on local machine)
    - auxdata (Data base of auxiliary datasets)
    """

    def __init__(self, package_root_dir: Path, version: str) -> None:
        """
        Collect package configuration data from the various definition files and provide an interface
        to pysiral processor, output and grid definition files.
        This class is intended to be only called inside the init module of pysiral and to store the
        pysiral package configuration in the global variable `psrlcfg`
        """

        # --- Establish the path information ---
        # This step gets the default path (user home, set path for the resources)
        # NOTE: The current path to the active pysiral package is already set in the global
        #       variable `pysiral.PACKAGE_ROOT_DIR`, since this is required for reading the
        #       version file
        self._package_root_dir = package_root_dir
        self._version = version

        # Get system and package information (needed for config files)
        self._system = self._get_system_config()
        self._package = self._get_package_config()

        # Read the configuration files (package configuration needed)
        self._platforms = self._get_platform_config()
        self._auxdata = self._get_auxdata_config()
        self._local_path = self._get_local_machine_config()

        # Create a catalog of processor and output definition files
        # (catalog only, validation of files is done on demand)
        self._procdef = self._get_procdef_catalog()
        self._outputdef = self._get_outputdef_catalog()

    @staticmethod
    def _get_system_config() -> SystemConfig:
        """
        Collect the system information and return as SystemConfig data model

        :return: SystemConfig data model
        """
        system_info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'hostname': socket.gethostname(),
            'processor': platform.processor(),
            'memory_gb': round(psutil.virtual_memory().total / (1024.0 ** 3)),
            'mp_cpu_count': multiprocessing.cpu_count(),
            "python_version": sys.version
        }
        return SystemConfig(**system_info)

    def _get_package_config(self) -> PackageConfig:
        """
        Collect the package information and return as PysiralPackageConfig data model

        :return: PysiralPackageConfig data model
        """

        cfg_loc_file = self._package_root_dir / "PYSIRAL-CFG-LOC"
        try:
            with open(str(cfg_loc_file)) as fh:
                config_target = fh.read().strip()
        except IOError:
            sys.exit(f"Cannot find PYSIRAL-CFG-LOC file in package (expected: {cfg_loc_file})")

        package_info = {
            'package_root': self._package_root_dir,
            'package_config': self._package_root_dir / "resources" / "pysiral-cfg",
            'user_home': Path.home(),
            'config_target': config_target
        }
        return PackageConfig(**package_info)

    def _get_platform_config(self) -> Union[PlatformConfig, None]:
        """
        Get the pysiral radar altimeter platform configuration ('mission_def.yaml') data model
        if the pysiral configuration is already configured.

        :return:
        """
        lookup_dir = self._package.package_config / "missions"
        if not lookup_dir.is_dir():
            return None

        # Get all yaml files
        yaml_files = sorted(list(lookup_dir.glob("*.yaml")))
        platforms_dict = {}
        for yaml_filepath in yaml_files:
            platform_id = yaml_filepath.stem
            content_dict = self._get_yaml_file_raw_dict(yaml_filepath)
            content_dict["filepath"] = yaml_filepath
            platforms_dict[platform_id] = content_dict

        return PlatformConfig(**platforms_dict)

    def _get_local_machine_config(self) -> Union[LocalMachineConfig, None]:
        """
        Return the data model for `local_machine_def.yaml` if the
        file has been configured, otherwise return None

        :return: Local Machine Def data model
        """

        if (filepath := self._package.local_machine_def_filepath) is None:
            return None

        content_dict = self._get_yaml_file_raw_dict(filepath)
        content_dict["filepath"] = filepath
        return LocalMachineConfig(**content_dict)

    def _get_auxdata_config(self) -> Union[AuxiliaryDataConfig, None]:
        """
        Return the data model for `auxdata_def.yaml` if the
        file has been configured, otherwise return None

        :return: Local Machine Def data model
        """
        filepath = self._package.package_config / _DEFINITION_FILES["auxdata"]
        if not filepath.is_file():
            return None

        content_dict = self._get_yaml_file_raw_dict(filepath)
        data_dict = {
            "filepath": filepath,
            "types": content_dict
        }
        return AuxiliaryDataConfig(**data_dict)

    def _get_procdef_catalog(self) -> Union[ProcDefCatalog, None]:
        """
        Create a catolog of processor definition yaml files in
        the pysiral configuration path (if known)
        :return:
        """
        dir_path = self._package.package_config / "proc"
        if not dir_path.is_dir():
            return None
        proc_levels = VALID_PROCESSOR_LEVELS
        return ProcDefCatalog(**self._get_yaml_ctlg_dict(dir_path, proc_levels))

    def _get_outputdef_catalog(self) -> Union[OutputDefCatalog, None]:
        """
        Create a catolog of processor definition yaml files in
        the pysiral configuration path (if known)
        :return:
        """
        dir_path = self._package.package_config / "output"
        if not dir_path.is_dir():
            return None

        # Only output definitions for l2i, l2p, and l3c
        proc_levels = list(VALID_DATA_LEVEL_IDS)
        proc_levels.remove("l1")
        proc_levels.remove("l2")
        proc_levels.remove(None)
        return OutputDefCatalog(**self._get_yaml_ctlg_dict(dir_path, proc_levels))

    @staticmethod
    def _get_yaml_ctlg_dict(dir_path: Path, proc_levels: List[str]) -> Dict:
        """
        Get a list of yaml files in all subdirectories of dirpath
        and return a dictionary with format needed for
        pysiral._package_config._YamlDefEntry

        :param dir_path: The lookup directory

        :return: dictionary catalog
        """
        ctlg_dict = {}
        for proc_level in proc_levels:
            lookup_dir = dir_path / proc_level
            yaml_files = sorted(list(lookup_dir.rglob("*.yaml")))
            ids = [y.stem for y in yaml_files]
            yaml_def_entries = [{"id": i, "filepath": f} for i, f in zip(ids, yaml_files)]
            ctlg_dict[proc_level] = dict(zip(ids, yaml_def_entries))
        return ctlg_dict

    @property
    def system(self) -> SystemConfig:
        return self._system.model_copy()

    @property
    def package(self) -> PackageConfig:
        return self._package.model_copy()

    @property
    def platforms(self) -> Union[PlatformConfig, None]:
        return self._platforms.model_copy()

    @property
    def auxdata(self) -> Union[AuxiliaryDataConfig, None]:
        return self._auxdata.model_copy()

    @property
    def local_path(self) -> Union[LocalMachineConfig, None]:
        return self._local_path.model_copy()

    @property
    def procdef(self) -> Union[ProcDefCatalog, None]:
        return self._procdef.model_copy()

    @property
    def outputdef(self) -> Union[OutputDefCatalog, None]:
        return self._outputdef.model_copy()

    @staticmethod
    def _get_yaml_file_raw_dict(filepath: Path) -> Dict:
        """
        Get a yaml file as raw dict. This is intended to be used
        if the content of a file are to be amended before passing
        to a pydantic model)

        :param filepath: Path to yaml file

        :return: Dictionary with yaml file content
        """
        reader = YAML(typ="safe", pure=True)  # YAML 1.2 support
        with filepath.open() as f:
            content_dict = reader.load(f)
        return content_dict
