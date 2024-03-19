# -*- coding: utf-8 -*-

"""
Module with configuration classes for the Level-1 PreProcessor
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import contextlib
import importlib
from pathlib import Path
from typing import Dict, List, Literal, Union, Optional
from pydantic import BaseModel, FilePath, PositiveInt, field_validator, model_validator, Field
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


class L1POutputHandlerConfig(BaseModel):
    options: Dict


class L1PProcPolarOceanConfig(BaseModel):
    target_hemisphere: List[Literal["north", "south"]]
    polar_latitude_threshold: float
    input_file_is_single_hemisphere: bool
    allow_nonocean_segment_nrecords: PositiveInt


class L1PProcOrbitConnectConfig(BaseModel):
    max_connected_segment_timedelta_seconds: PositiveInt


class L1PProcItemConfig(BaseModel):
    label: str
    stage: Literal["post_source_file", "post_ocean_segment_extraction", "post_merge"]
    module_name: str
    class_name: str
    options: Dict


class L1PProcOptions(BaseModel):
    polar_ocean: L1PProcPolarOceanConfig
    orbit_segment_connectivity: L1PProcOrbitConnectConfig
    processing_items: List[L1PProcItemConfig]


class L1PProcConfig(BaseModel):
    type: Literal["custom_orbit_segment", "half_orbit", "full_orbit"]
    options: L1PProcOptions


class L1pProcessorConfig(BaseModel):
    """
    Configuration data for the Level-1 pre-processor
    """
    filepath: Optional[FilePath] = Field(description="Filepath to the Level-1 pre-processor configuration object")
    supported_datasets: Union[str, List[str]] = Field(description="Supported datasets")
    dataset: Optional[str] = Field(description="Target platform", default=None, validate_default=False)
    input_handler: ClassConfig
    input_adapter: ClassConfig
    output_handler: L1POutputHandlerConfig
    level1_preprocessor: L1PProcConfig

    @classmethod
    def from_yaml(
            cls,
            filename_or_proc_id: Union[str, Path],
            input_dataset_id: str = None,
    ) -> "L1pProcessorConfig":
        """
        Initialize the class from the Level-1 pre-processor definition (yaml) file.


        :param filename_or_proc_id: A file id (must be known to the pysiral package configuration) or
            filepath to
        :param input_dataset_id: (Optional) input data set id for the target dataset.
            This parameter is needed for Level-1 pre-processor definitions that support multiple
            datasets.

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

        # Set filepath
        content_dict["filepath"] = config_filepath

        # Set target platform
        # (needed for Level-1 processor definitions that support multiple platforms)
        if input_dataset_id is not None:
            content_dict["dataset"] = input_dataset_id

        return cls(**content_dict)

    @property
    def supports_multiple_platforms(self) -> bool:
        return isinstance(self.supported_platforms, list)

    @field_validator("supported_datasets", "dataset")
    @classmethod
    def platform_must_be_known(cls, dataset_id):
        """
        Ensures that the tag `supported_platforms` in the L1 processor definition file
        only contains platform id's known to pysiral.

        :param dataset_id: str of list of string that should contain pysiral dataset id's

        :raises AssertionError: Invalid platform id

        :return: Validated supported_platforms
        """
        err_msg = f"Non pysiral-recognized datasets(s): {dataset_id=} {psrlcfg.platforms.dataset_id}"
        if isinstance(dataset_id, str):
            assert dataset_id in psrlcfg.platforms.ids, err_msg
        elif isinstance(dataset_id, list):
            assert all(p in psrlcfg.platforms.ids for p in dataset_id), err_msg
        # This shouldn't happen (to be caught by pydantic type validation)
        else: 
            raise TypeError(f"Invalid type {dataset_id=} {type(dataset_id)}")
        return dataset_id

    @model_validator(mode="after")
    def target_platform_must_be_specified(self):
        """
        Ensure that the platform field is always set. This field is not part of the
        Level-1 pre-processor definition file, which only contains the supported platforms.

        If this field is empty this method will set it with the supported platform, but
        raise a ValueError when multiple platforms are supported (ambiguous).

        :raise ValueError: platform field is empty and Level-1 pre-processor definition
            supports multiple platforms
        """
        print("target_platform_must_be_specified")
        if self.platform is None:
            if self.supports_multiple_platforms:
                raise ValueError(
                    f"{self.filepath} supports multiple platforms{self.supported_platforms}, "
                    "but target platform not specified"
                )
            else:
                self.platform = self.supported_platforms
        return self

    @model_validator(mode="after")
    def pass_down_platform(self):
        """
        Ensure that the input handler and adap
        """
        print("pass_down_platform")
        self.input_handler.options["platform"] = self.platform
        return self
