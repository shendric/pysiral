# -*- coding: utf-8 -*-
"""
This module contains the pydantic data model for the Level-1 processor configuration files
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = ["L1PreProcessorDefinition"]

from pydantic import BaseModel
from typing import List, Union, Dict, Any


class OutputHandlerConfig(BaseModel):
    """
    Data model for the output handler configuration in a Level-1 processor definition file
    """
    options: dict


class DataLoaderConfig(BaseModel):
    """
    Data model for the data loader configuration in a Level-1 processor definition file
    """
    class_name: str
    options: dict


class FileDiscoveryConfig(BaseModel):
    """
    Data model for the file discovery configuration in a Level-1 processor definition file
    """
    class_name: str
    options: dict


class AlgorithmConfig(BaseModel):
    """
    Data model for the algorithm configuration in a Level-1 processor definition file
    """
    label: str
    stage: str
    class_name: str
    options: Dict[str, Any]


class ProcessorOptions(BaseModel):
    """
    Data model for the processor options in a Level-1 processor definition file
    """
    polar_ocean: Dict[str, Any]
    discontinuities: Dict[str, Any]
    connectivity: Dict[str, Any]


class ProcessorConfig(BaseModel):
    type: str
    options: ProcessorOptions
    algorithms: List[AlgorithmConfig]


class L1PreProcessorDefinition(BaseModel):
    """
    Data model for the content of a Level-1 processor definition file (pysiral-cfg/proc-l1/**/*.yaml)
    """
    platforms: List[str]
    file_discovery: FileDiscoveryConfig
    data_loader: DataLoaderConfig
    output_handler: OutputHandlerConfig
    processor: ProcessorConfig
