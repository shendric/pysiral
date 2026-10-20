# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = ["EnvisatFDR4ALTFileDiscovery"]

from pydantic import BaseModel
from typing import Dict, Optional, List
from pathlib import Path

from pysiral.mission._io_abc import FileDiscovery


class EnvisatFDR4ALTFileDiscoveryConfig(BaseModel):
    local_machine_def_tag: str
    subtag: str
    lookup_dir: Dict[str, Path]


class EnvisatFDR4ALTFileDiscovery(FileDiscovery):

    def __init__(self, **kwargs):
        self.cfg = EnvisatFDR4ALTFileDiscoveryConfig(**kwargs)

    def query_period(self, period) -> Optional[List[Path]]:
        """
        Abstract method to load source data into Level 1 data object from a given file path.
        """
        pass
