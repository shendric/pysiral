# -*- coding: utf-8 -*-

"""
Configuration data model for the mission/platform definition files
(pysiral/resources/pysiral-cfg/mission/*.yaml)
"""

from datetime import datetime
from typing import Union, Dict, List
from pydantic import BaseModel, FilePath, ConfigDict
from ._models import ConvenientRootModel


class _AltimeterModeFlags(ConvenientRootModel):
    root: Dict[str, int]


class _AltimeterTimeCoverage(BaseModel):
    start: datetime
    end: Union[None, datetime]


class _AltimeterSourceDataset(BaseModel):
    long_name: str
    reference: str
    file_discovery_options: dict
    l1_input_adapter_options: dict


class _AltimeterSourceDatasets(ConvenientRootModel):
    root: Dict[str, _AltimeterSourceDataset]


class _AltimeterSensor(BaseModel):
    name: str
    modes: Dict


class _AltimeterPlatform(BaseModel):
    name: str
    long_name: str
    sensor: _AltimeterSensor
    time_coverage: _AltimeterTimeCoverage
    sea_ice_radar_modes: List[str]
    orbit_max_latitude: float
    source_datasets: _AltimeterSourceDatasets


class _AltimeterPlatforms(ConvenientRootModel):
    root: Dict[str, _AltimeterPlatform]


class _MissionDefinition(BaseModel):
    model_config = ConfigDict(extra="ignore")
    filepath: FilePath
    platforms: _AltimeterPlatforms


class MissionConfig(ConvenientRootModel):
    root: Dict[str, _MissionDefinition]

    @property
    def mission_ids(self) -> List[str]:
        return list(self.items)

    @property
    def platform_ids(self) -> List[str]:
        return sorted(self.platform_mission_dict.keys())

    @property
    def platform_mission_dict(self) -> Dict:
        platform_mission_dict = {}
        for mission_id in self.items:
            for platform in self[mission_id].platforms.items:
                platform_mission_dict[platform] = mission_id
        return platform_mission_dict
