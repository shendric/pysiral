# -*- coding: utf-8 -*-

"""
Module with configuration classes for the Level-1 PreProcessor
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import contextlib
import importlib
from argparse import Namespace
from typing import Dict, List, Literal
from pydantic import BaseModel, PositiveInt, field_validator

from pysiral import psrlcfg


class ClassConfig(BaseModel):
    module_name: str
    class_name: str
    options: Dict

    @classmethod
    @field_validator("module_name")
    def valid_pysiral_module(cls, module_name):
        # noinspection PyUnusedLocal
        result = None
        with contextlib.suppress(ImportError):
            result = importlib.import_module(module_name)
        assert result is not None, f"{module_name=} cannot be imported"
        return module_name


class L1POutputHandlerConfig(BaseModel):
    option: Dict


class L1PProcPolarOceanConfig(BaseModel):
    target_hemisphere: List[Literal["north", "south"]]
    polar_latitude_threshold: float
    input_file_is_single_hemisphere: bool
    allow_nonocean_segment_nrecords: PositiveInt


class L1PProcOrbitConnectConfig(BaseModel):
    max_connected_segment_timedelta_seconds: PositiveInt


class L1PProcItemConfig(BaseModel):
    label: str
    stage: Literal["post_source_file", "post_polar_ocean_segment_extraction", "post_merge"]
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
    platform: str
    input_handler: ClassConfig
    input_adapter: ClassConfig
    output_handler: L1POutputHandlerConfig
    level1_preprocessor: L1PProcConfig

    @classmethod
    def from_yaml(cls, ) -> "L1pProcessorConfig":
        """
        Initialize the

        :param args: The parameter from command line argumens
        :return:
        """

        breakpoint()

    @classmethod
    @field_validator("platform")
    def must_be_valid_platform(cls, platform):
        assert platform in psrlcfg.platforms.ids, f"Not pysiral-recognized {platform=} [{psrlcfg.platforms.ids}]"
        return platform

