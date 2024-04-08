# -*- coding: utf-8 -*-

"""
Module with configuration classes for the Level-1 PreProcessor
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import contextlib
import importlib
from pathlib import Path
from typing import Dict, List, Literal, Union, Optional, Any
from pydantic import BaseModel, FilePath, PositiveInt, field_validator, PositiveFloat, Field, ConfigDict, NonNegativeInt
from ruamel.yaml import YAML

from pysiral import psrlcfg
from pysiral.core.config import DataVersion


class ClassConfig(BaseModel):
    module_name: str
    class_name: str
    options: Dict

    @field_validator("module_name")
    @classmethod
    def valid_pysiral_module(cls, module_name):
        # noinspection PyUnusedLocal
        result = None
        with contextlib.suppress(ImportError):
            result = importlib.import_module(module_name)
        assert result is not None, f"{module_name=} cannot be imported"
        return module_name


class PolarOceanSegmentsConfig(BaseModel):
    orbit_coverage: Literal["custom_orbit_segment", "half_orbit", "full_orbit"]
    target_hemisphere: List[Literal["nh", "sh"]] = Field(default=["nh", "sh"])
    polar_latitude_threshold: PositiveFloat = 45.0,
    allow_nonocean_segment_nrecords: PositiveInt = 1000
    ocean_mininum_size_nrecords: PositiveInt = None
    timestamp_discontinuities: bool = False


class OrbitConnectConfig(BaseModel):
    max_timedelta_seconds: NonNegativeInt = Field(
        description="Acceptable gap in seconds between two connected segments"
    )


class ProcItemConfig(BaseModel):
    label: str
    stage: Literal["post_source_file", "post_ocean_segment_extraction", "post_merge"]
    class_name: str
    options: Dict


class PysiralPackageConfig(BaseModel):
    l1p_id: str = Field(description="Configuration id (must be unique for source data set")
    l1p_version: float = Field(description="Version number of the Level-1 processor definition")
    supported_source_datasets: List[str] = Field(description="List with id's of supported source data sets")

    @field_validator("supported_source_datasets")
    @classmethod
    def platform_must_be_known(cls, dataset_id):
        """
        Ensures that the tag `supported_platforms` in the L1 processor definition file
        only contains platform id's known to pysiral.

        :param dataset_id: str of list of string that should contain pysiral dataset id's

        :raises AssertionError: Invalid platform id

        :return: Validated supported_platforms
        """
        valid_ids = psrlcfg.missions.get_source_dataset_ids()
        err_msg = f"Non pysiral-recognized datasets(s): {dataset_id=} {valid_ids}"
        if isinstance(dataset_id, str):
            assert dataset_id in valid_ids, err_msg
        elif isinstance(dataset_id, list):
            assert all(p in valid_ids for p in dataset_id), err_msg
        # This shouldn't happen (to be caught by pydantic type validation)
        else:
            raise TypeError(f"Invalid type {dataset_id=} {type(dataset_id)}")
        return dataset_id


class Level1OutputHanderConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    l1p_id: str
    l1p_version: Union[str, float, DataVersion]
    minimum_n_records: NonNegativeInt = 0
    filename_template: Optional[str] = Field(
        description="filename template for l1p files",
        default=psrlcfg.local_path.pysiral_output.filenaming.l1p
    )
    filepath_template: Optional[str] = Field(
        description="subfolder filepath template for l1p files",
        default=psrlcfg.local_path.pysiral_output.sub_directories.l1p
    )

    @field_validator("l1p_version")
    @classmethod
    def is_dataversion(cls, value):
        return DataVersion(value) if isinstance(value, (str, float)) else value


class L1pProcessorConfig(BaseModel):
    """
    Configuration data for the Level-1 pre-processor
    """
    filepath: Optional[FilePath] = Field(
        description="Filepath to the Level-1 pre-processor configuration object"
    )
    pysiral_package_config: PysiralPackageConfig = Field(
        description="Information for pysiral package configuration"
    )
    source_data_discovery: Optional[Dict[str, Any]] = Field(
        description="Optional keyword arguments for source data discovery class",
        default={}
    )
    source_data_loader: Optional[Dict[str, Any]] = Field(
        description="Optional keyword arguments for source data loader class",
        default={}
    )
    polar_ocean_detection: PolarOceanSegmentsConfig = Field(
        description="Settings for retrieving polar ocean segments from source data"
    )
    orbit_segment_connect: OrbitConnectConfig = Field(
        description="Settings for connecting orbit segments"
    )
    output: Level1OutputHanderConfig = Field(
        description="Settings for writing l1p output files"
    )
    processing_items: List[ProcItemConfig] = Field(
        description="List of Level-1 pre-processor items"
    )

    @classmethod
    def from_yaml(
            cls,
            filename_or_proc_id: Union[str, Path],
            ) -> "L1pProcessorConfig":
        """
        Initialize the class from the Level-1 pre-processor definition (yaml) file.

        :param filename_or_proc_id: A file id (must be known to the pysiral package configuration) or
            filepath to

        :return: Initialized instance
        """

        # Resolve the filename
        if isinstance(filename_or_proc_id, Path):
            config_filepath = filename_or_proc_id
        else:
            config_filepath = psrlcfg.procdef.get("l1", filename_or_proc_id)

        # Read the file content to a raw dictionary
        # TODO: boilerpolate, move to function
        reader = YAML(typ="safe", pure=True)
        with config_filepath.open() as f:
            content_dict = reader.load(f)

        # Add filepath
        content_dict["filepath"] = config_filepath

        return cls(**content_dict)

    @property
    def supports_multiple_platforms(self) -> bool:
        return isinstance(self.pysiral_package_config.supported_source_datasets, list)
