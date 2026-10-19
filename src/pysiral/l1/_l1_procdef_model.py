# -*- coding: utf-8 -*-
"""
This module contains the pydantic data model for the
Level-1 processor configuration files
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = ["L1ProcessorDefinition"]

from pydantic import BaseModel


class FileDiscoveryConfig(BaseModel):
    """
    Data model for the file discovery configuration in a Level-1 processor definition file
    """
    class_name: str
    options: dict

class L1ProcessorDefinition(BaseModel):
    """
    Data model for the content of a Level-1 processor definition file (pysiral-cfg/proc-l1/**/*.yaml)
    """
    platforms: Union[List[str], str]
    file_discovery: FileDiscoveryConfig
    data_loader: DataLoaderConfig
    output_handler: OutputHandlerConfig
    processor: ProcessorConfig

