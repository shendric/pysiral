# -*- coding: utf-8 -*-

"""
Generic data model for the pysiral package configuration
"""


from typing import Dict, List
from pydantic import RootModel, BaseModel, FilePath


class _YamlDefEntry(BaseModel):
    id: str
    filepath: FilePath


class ConvenientRootModel(RootModel):
    """
    A helper class for yaml configuraton structures with variable key.
    """
    root: Dict[str, Dict]

    @property
    def items(self) -> List[str]:
        return sorted(list(self.root.keys()))

    def __getattr__(self, item):
        return self.root[item]

    def __getitem__(self, item):
        return self.root[item]

    def __contains__(self, item):
        return item in self.items
