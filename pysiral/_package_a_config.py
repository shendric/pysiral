# import multiprocessing
# import socket
# import platform
# import psutil
# import sys
# from pathlib import Path
# from typing import Union, Dict, List, Literal, Optional
# from pydantic import BaseModel, FilePath, PositiveInt, PositiveFloat
# from ruamel.yaml import YAML
#
#
# # Filenames of definitions files
# _DEFINITION_FILES = {
#     "auxdata": "auxdata_def.yaml",
# }
#
# # name of the file containing the data path on the local machine
# _LOCAL_MACHINE_DEF_FILE = "local_machine_def.yaml"
#
# # valid settings types, processor levels and data level ids.
# # NOTE: These tie into the naming and content of definition files
# VALID_SETTING_TYPES = ["proc", "output", "grid"]
# VALID_PROCESSOR_LEVELS = ["l1", "l2", "l3"]
# VALID_DATA_LEVEL_IDS = ["l1", "l2", "l2i", "l2p", "l3", None]
# VALID_CONFIG_TARGETS = ["PACKAGE", "USER_HOME"]
#
#
# class _YamlDefEntry(BaseModel):
#     id: str
#     filepath: FilePath
#
#
# class ProcDefCatalog(BaseModel):
#     l1: Dict[str, _YamlDefEntry]
#     l2: Dict[str, _YamlDefEntry]
#     l3: Dict[str, _YamlDefEntry]
#
#     def get(
#             self,
#             proc_level: Literal["l1", "l2", "l3"],
#             yaml_id: str,
#             raise_if_none: bool = False
#     ) -> Optional[Path]:
#         """
#         Return the file path for processor definition config base from the
#         processing level and yaml file id.
#
#         :param proc_level: A valid processor definition (l1, l2, l3)
#         :param yaml_id: The id of the yaml file (filepath.setm)
#         :param raise_if_none: Boolean flag defining action if yaml file does not exist
#
#         :raises KeyError: `proc_level` not a valid processing level for processor
#                 definition files.
#
#         :raises FileNotFoundError: Target yaml file does not exist and `raise_if_none=True`
#
#         :return: filepath or None
#
#         """
#         if proc_level not in self.model_fields:
#             if raise_if_none:
#                 raise KeyError(f"{proc_level=} not a valid processor level [l1, l2, l3]")
#             return None
#
#         yaml_def_entry = getattr(self, proc_level).get(yaml_id)
#         if yaml_def_entry is None and raise_if_none:
#             raise FileNotFoundError(f"No file found for {proc_level}:{yaml_id}")
#         elif yaml_def_entry is None:
#             return None
#         return yaml_def_entry.filepath
#
#
# class OutputDefCatalog(BaseModel):
#     l2i: Dict[str, _YamlDefEntry]
#     l2p: Dict[str, _YamlDefEntry]
#     l3: Dict[str, _YamlDefEntry]
#
#
# class _PysiralPackageConfiguration(object):
#     """
#     Container for the content of the pysiral definition files
#     (in pysiral/configuration) and the local machine definition file
#     (local_machine_definition.yaml)
#
#     Main Properties
#     - system (system properties)
#     - package (pysiral package information, e.g. config path etc.)
#     - platforms (information on supported platforms)
#     - local_path (Path to data on local machine)
#     - auxdata (Data base of auxiliary datasets)
#     """
#
#     def __init__(self, package_root_dir: Path, version: str) -> None:
#         """
#         Collect package configuration data from the various definition files and provide an interface
#         to pysiral processor, output and grid definition files.
#         This class is intended to be only called inside the init module of pysiral and to store the
#         pysiral package configuration in the global variable `psrlcfg`
#         """
#
#         # --- Establish the path information ---
#         # This step gets the default path (user home, set path for the resources)
#         # NOTE: The current path to the active pysiral package is already set in the global
#         #       variable `pysiral.PACKAGE_ROOT_DIR`, since this is required for reading the
#         #       version file
#         self._package_root_dir = package_root_dir
#         self._version = version
#
#         # Get system and package information (needed for config files)
#         self._system = self._get_system_config()
#         self._package = self._get_package_config()
#
#         # Read the configuration files (package configuration needed)
#         self._platforms = self._get_platform_config()
#         self._auxdata = self._get_auxdata_config()
#         self._local_path = self._get_local_machine_config()
#
#         # Create a catalog of processor and output definition files
#         # (catalog only, validation of files is done on demand)
#         self._procdef = self._get_procdef_catalog()
#         self._outputdef = self._get_outputdef_catalog()
#
#     @staticmethod
#     def _get_system_config() -> SystemConfig:
#         """
#         Collect the system information and return as SystemConfig data model
#
#         :return: SystemConfig data model
#         """
#         system_info = {
#             'platform': platform.system(),
#             'platform_release': platform.release(),
#             'platform_version': platform.version(),
#             'architecture': platform.machine(),
#             'hostname': socket.gethostname(),
#             'processor': platform.processor(),
#             'memory_gb': round(psutil.virtual_memory().total / (1024.0 ** 3)),
#             'mp_cpu_count': multiprocessing.cpu_count(),
#             "python_version": sys.version
#         }
#         return SystemConfig(**system_info)
#
#     def _get_package_config(self) -> PackageConfig:
#         """
#         Collect the package information and return as PysiralPackageConfig data model
#
#         :return: PysiralPackageConfig data model
#         """
#
#         cfg_loc_file = self._package_root_dir / "PYSIRAL-CFG-LOC"
#         try:
#             with open(str(cfg_loc_file)) as fh:
#                 config_target = fh.read().strip()
#         except IOError:
#             sys.exit(f"Cannot find PYSIRAL-CFG-LOC file in package (expected: {cfg_loc_file})")
#
#         package_info = {
#             'package_root': self._package_root_dir,
#             'package_config': self._package_root_dir / "resources" / "pysiral-cfg",
#             'user_home': Path.home(),
#             'config_target': config_target
#         }
#         return PackageConfig(**package_info)
#
#     def _get_platform_config(self) -> Union[PlatformConfig, None]:
#         """
#         Get the pysiral radar altimeter platform configuration ('mission_def.yaml') data model
#         if the pysiral configuration is already configured.
#
#         :return:
#         """
#         lookup_dir = self._package.package_config / "missions"
#         if not lookup_dir.is_dir():
#             return None
#
#         # Get all yaml files
#         yaml_files = sorted(list(lookup_dir.glob("*.yaml")))
#         platforms_dict = {}
#         for yaml_filepath in yaml_files:
#             platform_id = yaml_filepath.stem
#             content_dict = self._get_yaml_file_raw_dict(yaml_filepath)
#             content_dict["filepath"] = yaml_filepath
#             platforms_dict[platform_id] = content_dict
#
#         return PlatformConfig(**platforms_dict)
#
#     def _get_local_machine_config(self) -> Union[LocalMachineConfig, None]:
#         """
#         Return the data model for `local_machine_def.yaml` if the
#         file has been configured, otherwise return None
#
#         :return: Local Machine Def data model
#         """
#
#         if (filepath := self._package.local_machine_def_filepath) is None:
#             return None
#
#         content_dict = self._get_yaml_file_raw_dict(filepath)
#         content_dict["filepath"] = filepath
#         return LocalMachineConfig(**content_dict)
#
#     def _get_auxdata_config(self) -> Union[AuxiliaryDataConfig, None]:
#         """
#         Return the data model for `auxdata_def.yaml` if the
#         file has been configured, otherwise return None
#
#         :return: Local Machine Def data model
#         """
#         filepath = self._package.package_config / _DEFINITION_FILES["auxdata"]
#         if not filepath.is_file():
#             return None
#
#         content_dict = self._get_yaml_file_raw_dict(filepath)
#         data_dict = {
#             "filepath": filepath,
#             "types": content_dict
#         }
#         return AuxiliaryDataConfig(**data_dict)
#
#     def _get_procdef_catalog(self) -> Union[ProcDefCatalog, None]:
#         """
#         Create a catolog of processor definition yaml files in
#         the pysiral configuration path (if known)
#         :return:
#         """
#         dir_path = self._package.package_config / "proc"
#         if not dir_path.is_dir():
#             return None
#         proc_levels = VALID_PROCESSOR_LEVELS
#         return ProcDefCatalog(**self._get_yaml_ctlg_dict(dir_path, proc_levels))
#
#     def _get_outputdef_catalog(self) -> Union[OutputDefCatalog, None]:
#         """
#         Create a catolog of processor definition yaml files in
#         the pysiral configuration path (if known)
#         :return:
#         """
#         dir_path = self._package.package_config / "output"
#         if not dir_path.is_dir():
#             return None
#
#         # Only output definitions for l2i, l2p, and l3c
#         proc_levels = list(VALID_DATA_LEVEL_IDS)
#         proc_levels.remove("l1")
#         proc_levels.remove("l2")
#         proc_levels.remove(None)
#         return OutputDefCatalog(**self._get_yaml_ctlg_dict(dir_path, proc_levels))
#
#     @staticmethod
#     def _get_yaml_ctlg_dict(dir_path: Path, proc_levels: List[str]) -> Dict:
#         """
#         Get a list of yaml files in all subdirectories of dirpath
#         and return a dictionary with format needed for
#         pysiral._package_config._YamlDefEntry
#
#         :param dir_path: The lookup directory
#
#         :return: dictionary catalog
#         """
#         ctlg_dict = {}
#         for proc_level in proc_levels:
#             lookup_dir = dir_path / proc_level
#             yaml_files = sorted(list(lookup_dir.rglob("*.yaml")))
#             ids = [y.stem for y in yaml_files]
#             yaml_def_entries = [{"id": i, "filepath": f} for i, f in zip(ids, yaml_files)]
#             ctlg_dict[proc_level] = dict(zip(ids, yaml_def_entries))
#         return ctlg_dict
#
#     @property
#     def system(self) -> SystemConfig:
#         return self._system.model_copy()
#
#     @property
#     def package(self) -> PackageConfig:
#         return self._package.model_copy()
#
#     @property
#     def platforms(self) -> Union[PlatformConfig, None]:
#         return self._platforms.model_copy()
#
#     @property
#     def auxdata(self) -> Union[AuxiliaryDataConfig, None]:
#         return self._auxdata.model_copy()
#
#     @property
#     def local_path(self) -> Union[LocalMachineConfig, None]:
#         return self._local_path.model_copy()
#
#     @property
#     def procdef(self) -> Union[ProcDefCatalog, None]:
#         return self._procdef.model_copy()
#
#     @property
#     def outputdef(self) -> Union[OutputDefCatalog, None]:
#         return self._outputdef.model_copy()
#
#     # def _get_pysiral_path_information(self):
#     #     """
#     #     Get the different path information for pysiral. This method will add the following
#     #     attributes to self.path:
#     #         1. package_root_path: The root directory of this package
#     #         2. package_config_path: The directory of the pysiral config items in this package
#     #         3. userhome_config_dir: The intended configuration directory in the user home
#     #         4. config_target: The value given in the `PYSIRAL-CFG-LOC` file
#     #     :return: None
#     #     """
#     #
#     #     # Store the root dir of this pysiral package
#     #     self._path["package_root_path"] = self._package_root_dir
#     #
#     #     # Get the config directory of the package
#     #     # NOTE: This approach should work for a local script location or an installed package
#     #     self._path["package_config_path"] = self._path["package_root_path"] / "resources" / "pysiral-cfg"
#     #
#     #     # Get an indication of the location for the pysiral configuration path
#     #     # NOTE: In its default version, the text file `PYSIRAL-CFG-LOC` does only contain the
#     #     #       string `USER_HOME`. In this case, pysiral will expect the a .pysiral-cfg sub-folder
#     #     #       in the user home. The only other valid option is an absolute path to a specific
#     #     #       directory with the same content as .pysiral-cfg. This was introduced to enable
#     #     #       fully encapsulated pysiral installation in virtual environments
#     #
#     #     # Get the home directory of the current user
#     #     self._path["userhome_config_path"] = Path.home() / ".pysiral-cfg"
#     #
#     #     # Read pysiral config location indicator file
#     #     cfg_loc_file = self._package_root_dir / "PYSIRAL-CFG-LOC"
#     #     try:
#     #         with open(str(cfg_loc_file)) as fh:
#     #             self._path["config_target"] = fh.read().strip()
#     #     except IOError:
#     #         sys.exit(f"Cannot find PYSIRAL-CFG-LOC file in package (expected: {cfg_loc_file})")
#
#     # def _check_pysiral_config_path(self):
#     #     """
#     #     This class ensures that the pysiral configuration files are in the chosen
#     #     configuration directory
#     #     :return:
#     #     """
#     #
#     #     # Make alias of
#     #     config_path = Path(self.config_path)
#     #     package_config_path = Path(self.path.package_config_path)
#     #
#     #     # Check if current config dir is package config dir
#     #     # if yes -> nothing to do (files are either there or aren't)
#     #     if config_path == package_config_path:
#     #         return
#     #
#     #     # current config dir is not package dir and does not exist
#     #     # -> must be populated with content from the package config dir
#     #     if not config_path.is_dir():
#     #         print(f"Creating pysiral config directory: {config_path}")
#     #         dir_util.copy_tree(str(self.path.package_config_path), str(config_path))
#     #         print("Init local machine def")
#     #         template_filename = package_config_path / "templates" / "local_machine_def.yaml"
#     #         target_filename = config_path / "local_machine_def.yaml"
#     #         shutil.copy(str(template_filename), str(target_filename))
#
#     @staticmethod
#     def _get_yaml_file_raw_dict(filepath: Path) -> Dict:
#         """
#         Get a yaml file as raw dict. This is intended to be used
#         if the content of a file are to be amended before passing
#         to a pydantic model)
#
#         :param filepath: Path to yaml file
#
#         :return: Dictionary with yaml file content
#         """
#         reader = YAML(typ="safe", pure=True)  # YAML 1.2 support
#         with filepath.open() as f:
#             content_dict = reader.load(f)
#         return content_dict
#
#     # def _read_config_files(self):
#     #     """
#     #     Read the three main configuration files for
#     #         1. supported platforms
#     #         2. supported auxiliary datasets
#     #         3. path on local machine
#     #     and create the necessary catalogues
#     #     :return:
#     #     """
#     #
#     #     # --- Get information of supported platforms ---
#     #     # The general information for supported radar altimeter missions (mission_def.yaml)
#     #     # provides general metadata for each altimeter missions that can be used to sanity checks
#     #     # and queries for sensor names etc.
#     #     #
#     #     # NOTE: This is just general information on altimeter platform and not to be confused with
#     #     #       settings for actual primary data files. These are located in each l1p processor
#     #     #       definition file.
#     #     self.mission_def_filepath = self.config_path / Path(self._DEFINITION_FILES["platforms"])
#     #     if not self.mission_def_filepath.is_file():
#     #         error_msg = "Cannot load pysiral package files: \n %s" % self.mission_def_filepath
#     #         print(error_msg)
#     #         sys.exit(1)
#     #     self.platforms = _MissionDefinitionCatalogue(self.mission_def_filepath)
#     #
#     #     # --- Get information on supported auxiliary data sets ---
#     #     # The auxdata_def.yaml config file contains the central definition of the properties
#     #     # of supported auxiliary data sets. Each auxiliary data set is uniquely defined by
#     #     # the type of auxiliary data set and a name id.
#     #     # The central definition allows accessing auxiliary data by its id in processor definition files
#     #     self.auxdata_def_filepath = self.config_path / self._DEFINITION_FILES["auxdata"]
#     #     if not self.auxdata_def_filepath.is_file():
#     #         error_msg = "Cannot load pysiral package files: \n %s" % self.auxdata_def_filepath
#     #         print(error_msg)
#     #         sys.exit(1)
#     #     self.auxdef = _AuxdataCatalogue(self.auxdata_def_filepath)
#     #
#     #     # read the local machine definition file
#     #     self._read_local_machine_file()
#
#     # @staticmethod
#     # def get_yaml_config(filename):
#     #     """
#     #     Read a yaml file and return it content as an attribute-enabled dictionary
#     #     :param filename: path to the yaml file
#     #     :return: attrdict.AttrDict
#     #     """
#     #     with open(str(filename)) as fileobj:
#     #         settings = AttrDict(yaml.safe_load(fileobj))
#     #     return settings
#
#     # def get_setting_ids(self, settings_type, data_level=None):
#     #     lookup_directory = self.get_local_setting_path(settings_type, data_level)
#     #     ids, files = self.get_yaml_setting_filelist(lookup_directory)
#     #     return ids
#
#     # def get_platform_period(self, platform_id):
#     #     """
#     #     Get a period definition for a given platform ID
#     #     :param platform_id:
#     #     :return: dateperiods.DatePeriod
#     #     """
#     #     tcs, tce = self.platforms.get_time_coverage(platform_id)
#     #     return DatePeriod(tcs, tce)
#
#     # def get_processor_definition_ids(self, processor_level):
#     #     """
#     #     Returns a list of available processor definitions ids for a given processor
#     #     level (see self.VALID_PROCESSOR_LEVELS)
#     #     :param processor_level:
#     #     :return:
#     #     """
#     #     lookup_directory = self.get_local_setting_path("proc", processor_level)
#     #     return self.get_yaml_setting_filelist(lookup_directory, return_value="ids")
#
#     # def get_settings_files(self, settings_type: str, data_level: str) -> Iterable[Path]:
#     #     """
#     #     Returns all processor settings or output definitions files for a given data level.
#     #     :param settings_type:
#     #     :param data_level:
#     #     :return:
#     #     """
#     #
#     #     if settings_type not in VALID_SETTING_TYPES:
#     #         return []
#     #
#     #     if data_level not in VALID_DATA_LEVEL_IDS:
#     #         return []
#     #
#     #     # Get all settings files in settings/{data_level} and its
#     #     # subdirectories
#     #     lookup_directory = self.get_local_setting_path(settings_type, data_level)
#     #     _, files = self.get_yaml_setting_filelist(lookup_directory)
#     #
#     #     # Test if ids are unique and return error for the moment
#     #     return files
#
#     # def get_settings_file(self, settings_type, data_level, setting_id_or_filename):
#     #     """ Returns a processor settings file for a given data level.
#     #     (data level: l2 or l3). The second argument can either be an
#     #     direct filename (which validity will be checked) or an id, for
#     #     which the corresponding file (id.yaml) will be looked up in
#     #     the default directory """
#     #
#     #     if settings_type not in VALID_SETTING_TYPES:
#     #         return None
#     #
#     #     if data_level not in VALID_DATA_LEVEL_IDS:
#     #         return None
#     #
#     #     # Check if filename
#     #     if Path(setting_id_or_filename).is_file():
#     #         return setting_id_or_filename
#     #
#     #     # Get all settings files in settings/{data_level} and its
#     #     # subdirectories
#     #     lookup_directory = self.get_local_setting_path(settings_type, data_level)
#     #     ids, files = self.get_yaml_setting_filelist(lookup_directory)
#     #
#     #     # Test if ids are unique and return error for the moment
#     #     if len(set(ids)) != len(ids):
#     #         msg = f"Non-unique {settings_type}-{str(data_level)} setting filename"
#     #         print(f"ambiguous-setting-files: {msg}")
#     #         sys.exit(1)
#     #
#     #     # Find filename to setting_id
#     #     try:
#     #         index = ids.index(setting_id_or_filename)
#     #         return Path(files[index])
#     #     except (IOError, ValueError):
#     #         return None
#
#     # @staticmethod
#     # def get_yaml_setting_filelist(directory, return_value="both"):
#     #     """ Retrieve all yaml files from a given directory (including
#     #     subdirectories). Directories named "obsolete" are ignored if
#     #     ignore_obsolete=True (default) """
#     #     setting_ids = []
#     #     setting_files = []
#     #     for filepath in directory.rglob("*.yaml"):
#     #         setting_ids.append(filepath.name.replace(".yaml", ""))
#     #         setting_files.append(filepath)
#     #     if return_value == "both":
#     #         return setting_ids, setting_files
#     #     elif return_value == "ids":
#     #         return setting_ids
#     #     elif return_value == "files":
#     #         return setting_files
#     #     else:
#     #         raise ValueError(f"Unknown return value {str(return_value)} [`both`, `ids`, `files`]")
#
#     # def get_local_setting_path(self, settings_type, data_level=None):
#     #     """
#     #     Return the absolute path on the local productions system to the configuration file. The
#     #     returned path depends on the fixed structure below the `resources` directory in the pysiral
#     #     package and the choice in the config file "PYSIRAL-CFG-LOC"
#     #     :param settings_type:
#     #     :param data_level:
#     #     :return:
#     #     """
#     #     if settings_type in VALID_SETTING_TYPES and data_level in VALID_DATA_LEVEL_IDS:
#     #         args = [settings_type]
#     #         if data_level is not None:
#     #             args.append(data_level)
#     #         return Path(self.config_path) / Path(*args)
#     #     else:
#     #         return None
#
#     # def reload(self):
#     #     """
#     #     Method to trigger reading the configuration files again, e.g. after changing the config target
#     #     :return:
#     #     """
#     #     self._read_config_files()
#     #     self._check_pysiral_config_path()
#
#     # def set_config_target(self, config_target, permanent=False):
#     #     """
#     #     Set the configuration target
#     #     :param config_target:
#     #     :param permanent:
#     #     :return:
#     #     """
#     #
#     #     # Input validation
#     #     if config_target in VALID_CONFIG_TARGETS or Path(config_target).is_dir():
#     #         self._path["config_target"] = config_target
#     #     else:
#     #         msg = "Invalid config_target: {} must be {} or valid path"
#     #         msg = msg.format(str(config_target), ", ".join(VALID_CONFIG_TARGETS))
#     #         raise ValueError(msg)
#     #
#     #     if permanent:
#     #         raise NotImplementedError()
#
#     # def _read_local_machine_file(self) -> Union[LocalMachineConfig, None]:
#     #     """
#     #     Read the local machine definition file if it exists or return None
#     #     (necessary for automatic tests)
#     #
#     #     :return: The local_machine_def.yaml content as data model
#     #     """
#     #     filename = self.local_machine_def_filepath
#     #     if filename.is_file():
#     #         return parse_yaml_file_as(LocalMachineConfig, str(filename))
#     #
#     #     msg = f"local_machine_def.yaml not found (expected: {filename})"
#     #     logger.error(f"local-machine-def-missing: {msg}")
#     #     return None
#
#     # @property
#     # def platform_ids(self):
#     #     return self.platforms.ids
#     #
#     # @property
#     # def package_path(self) -> Path:
#     #     return Path(self._package_root_dir)
#     #
#     # @property
#     # def current_config_target(self):
#     #     return str(self._path["config_target"])
#     #
#     # @property
#     # def config_target(self):
#     #     return str(self._path["config_target"])
#     #
#     # @property
#     # def config_path(self):
#     #     """
#     #     nstruct the target config path based on the value in `PYSIRAL-CFG-LOC`
#     #     :return:
#     #     """
#     #     # Case 1 (default): pysiral config path is in user home
#     #     if self._path["config_target"] == "USER_HOME":
#     #         return Path(self._path["userhome_config_path"])
#     #
#     #     # Case 2: pysiral config path is the package itself
#     #     elif self._path["config_target"] == "PACKAGE":
#     #         return Path(self._path["package_config_path"])
#     #
#     #     # Case 3: package specific config path
#     #     else:
#     #         # This should be an existing path, but in the case it is not, it is created
#     #         return Path(self._path["config_target"])
