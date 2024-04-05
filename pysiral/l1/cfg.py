# -*- coding: utf-8 -*-

"""
Module with configuration classes for the Level-1 PreProcessor
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import contextlib
import importlib
from pathlib import Path
from typing import Dict, List, Literal, Union, Optional
from pydantic import BaseModel, FilePath, PositiveInt, field_validator, PositiveFloat, Field
from ruamel.yaml import YAML

from pysiral import psrlcfg


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


class L1PProcPolarOceanConfig(BaseModel):
    target_hemisphere: List[Literal["nh", "sh"]]
    polar_latitude_threshold: float
    input_file_is_single_hemisphere: bool
    allow_nonocean_segment_nrecords: PositiveInt


class L1PProcOrbitConnectConfig(BaseModel):
    max_connected_segment_timedelta_seconds: PositiveInt


class L1PProcItemConfig(BaseModel):
    label: str
    stage: Literal["post_source_file", "post_ocean_segment_extraction", "post_merge"]
    class_name: str
    options: Dict


class L1PProcConfig(BaseModel):
    polar_ocean: L1PProcPolarOceanConfig
    orbit_segment_connectivity: L1PProcOrbitConnectConfig
    processing_items: List[L1PProcItemConfig]


class L1PPysiralPackageConfig(BaseModel):
    id: str = Field(description="Configuration id (must be unique for source data set")
    version: float = Field(description="Version number of the Level-1 processor definition")
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


class L1pProcessorConfig(BaseModel):
    """
    Configuration data for the Level-1 pre-processor
    """
    filepath: Optional[FilePath] = Field(description="Filepath to the Level-1 pre-processor configuration object")
    pysiral_package_config: L1PPysiralPackageConfig
    level1_preprocessor: L1PProcConfig

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

        # Read the file content to a raw dictionay
        # TODO: boilerpolate, move to function
        reader = YAML(typ="safe", pure=True)
        with config_filepath.open() as f:
            content_dict = reader.load(f)

        # Add filepath
        content_dict["filepath"] = config_filepath

        return cls(**content_dict)

    @property
    def supports_multiple_platforms(self) -> bool:
        return isinstance(self.supported_platforms, list)


class PolarOceanSegmentsConfig(BaseModel):
    orbit_coverage: str = Literal["custom_orbit_segment", "half_orbit", "full_orbit"]
    target_hemisphere: List[Literal["nh", "sh"]] = Field(default=["nh", "sh"])
    polar_latitude_threshold: PositiveFloat = 45.0,
    allow_nonocean_segment_nrecords: PositiveInt = 1000
    ocean_mininum_size_nrecords: PositiveInt = None
