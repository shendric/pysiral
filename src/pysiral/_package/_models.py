# -*- coding: utf-8 -*-
"""
Generic data model for the pysiral package configuration
"""


__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


from typing import Dict, List
from pydantic import RootModel


class ConvenientRootModel(RootModel):
    """
    A helper class for dictionary-like structures with variable key(s).
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

