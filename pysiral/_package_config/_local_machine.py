# -*- coding: utf-8 -*-

"""
Configuration data model for the local path config file (local_machine_def.yaml)
"""

import re
from pathlib import Path
from typing import Union, Dict, List, Optional
from pydantic import BaseModel, DirectoryPath, FilePath, field_validator, ConfigDict, Field
from pysiral._package_config._models import ConvenientRootModel


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
    id: str
    path: Union[str, Dict]


class _RadarAltimeterCatalogPlatforms(ConvenientRootModel):
    root: Dict[str, List[_DataSourceEntry]]


class _PysiralOutputDirectory(BaseModel):
    base_directory: DirectoryPath
    sub_directories: _PysiralProductOutputPattern


class _AuxiliaryDataPath(BaseModel):
    auxtype: str
    sources: List[_DataSourceEntry]


class LocalMachineConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    filepath: FilePath = Field(description="Filepath of the configuration file")
    pysiral_output: _PysiralOutputDirectory
    radar_altimeter_catalog: _RadarAltimeterCatalogPlatforms
    auxiliary_data_catalog: List[_AuxiliaryDataPath]
    source_data_ctlg:  Dict = Field(
        description="Catalog of all source data set id's",
        default={}
    )

    def __init__(self, **data) -> None:
        """
        Create data catalogues after the

        :param data: The input to this class passed on to pydantic.BaseModel
        """
        super().__init__(**data)
        self._register_all_source_datasets()

    def _register_all_source_datasets(self):
        """
        Create a dictionary of all source data set id's
        """
        for platform in self.radar_altimeter_catalog.items:
            for entry in self.radar_altimeter_catalog[platform]:
                if entry.id in self.source_data_ctlg:
                    raise ValueError(f"source_dataset_id={entry.id} is defined more than once in {self.filepath}")
                self.source_data_ctlg[entry.id] = entry

    def get_source_directory(
            self,
            source_data_id: str,
            raise_if_none: bool = False
    ) -> Optional[Union[Path, Dict]]:
        """
        Return the source path(s) of a specific dataset from `local_machine_def.yaml`

        :param source_data_id: Must be valid sourcedata set identifier
        :param raise_if_none: Boolean flag defining action if yaml file does not exist

        :raises KeyError: Incorrect platform id

        :return: Either Path or Dict[Path]
        """

        if source_data_id not in self.source_data_ctlg and raise_if_none:
            raise KeyError(f"{source_data_id=} not a valid source dataset identifier [{self.platforms.items}]")

        # if (platform_def := self.platforms[platform_id]) is None:
        #     if raise_if_none:
        #         raise KeyError(f"{platform_id=} not a valid platform_id [{self.platforms.items}]")
        #     return None
        #
        # if (platform_source := platform_def[source_name]) is None:
        #     if raise_if_none:
        #         raise KeyError(f"No platform data definition {platform_id=}:{source_name=}")
        #     return None
        #
        # return platform_source.source
