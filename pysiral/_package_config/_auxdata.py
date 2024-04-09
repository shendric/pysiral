# -*- coding: utf-8 -*-

"""
Configuration data model for the auxiliary data definition (in auxdata_def.yaml)
"""

from typing import Union, Optional, List, Dict, Tuple
from pydantic import BaseModel, FilePath

from ._models import ConvenientRootModel


class _AuxDataDef(BaseModel):
    pyclass: str
    long_name: str
    local_repository: Union[str, None]
    filenaming: Optional[str] = None
    filename: Optional[str] = None
    options: Optional[Dict] = {}
    source: Optional[Union[Dict, None]] = {}
    sub_folders: Optional[List[str]] = []


class _AuxiliaryDataType(ConvenientRootModel):
    root: Dict[str, _AuxDataDef]


class _AuxiliaryDataTypes(ConvenientRootModel):
    root: Dict[str, _AuxiliaryDataType]


class AuxiliaryDataConfig(BaseModel):
    filepath: FilePath
    types: _AuxiliaryDataTypes

    def get_entries(self) -> List[Tuple[str, str, _AuxDataDef]]:
        output = []
        for auxdata_type in self.types.items:
            output.extend(
                (auxdata_type, entry, self.types[auxdata_type][entry])
                for entry in self.types[auxdata_type].items
            )
        return output
