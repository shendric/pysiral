# -*- coding: utf-8 -*-

"""
Configuration data model for the local path config file (local_machine_def.yaml)
"""

import re
from pathlib import Path
from typing import Union, Dict, List, Optional
from pydantic import BaseModel, DirectoryPath, FilePath, field_validator, ConfigDict


class _PysiralProductOutputPattern(BaseModel):
    l1p: str
    l2: str
    l3: str

    @field_validator("l1p", "l2", "l3")
    @classmethod
    def test_pysiral_output_dir_pattern(cls, pattern):
        regex = re.compile(r"[^\w\\/{}]")
        assert not regex.findall(pattern), f"regex failed: {pattern}"
        return pattern


class _DataSourceEntry(BaseModel):
    name: str
    path: Union[str, Dict]


class _RadarAltimeterCatalogPlatformEntry(BaseModel):
    platform: str
    default: str
    sources: List[_DataSourceEntry]


class _PysiralOutputDirectory(BaseModel):
    base_directory: DirectoryPath
    sub_directories: _PysiralProductOutputPattern


class _AuxiliaryDataPath(BaseModel):
    auxtype: str
    sources: List[_DataSourceEntry]


class LocalMachineConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    filepath: FilePath
    pysiral_output: _PysiralOutputDirectory
    radar_altimeter_catalog: List[_RadarAltimeterCatalogPlatformEntry]
    auxiliary_data_catalog: List[_AuxiliaryDataPath]

    def get_ra_source(
            self,
            platform_id: str,
            source_name: str,
            raise_if_none: bool = False
    ) -> Optional[Union[Path, Dict]]:
        """
        Return the source path(s) of a specific dataset from `local_machine_def.yaml`

        :param platform_id: Must be valid platform id
        :param source_name: Must be valid source name for platform id
        :param raise_if_none: Boolean flag defining action if yaml file does not exist

        :raises KeyError: Incorrect platform id
        :raises KeyError: Incorrect source name

        :return: Either Path or Dict[Path]
        """
        if (platform_def := self.platforms[platform_id]) is None:
            if raise_if_none:
                raise KeyError(f"{platform_id=} not a valid platform_id [{self.platforms.items}]")
            return None

        if (platform_source := platform_def[source_name]) is None:
            if raise_if_none:
                raise KeyError(f"No platform data definition {platform_id=}:{source_name=}")
            return None

        return platform_source.source
