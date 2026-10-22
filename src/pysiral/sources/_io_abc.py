# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
from dateperiods import DatePeriod

import pandas as pd

from pysiral import PSRLCFG
from pysiral.l1.l1_data import Level1bData


class SourceDataLoader(ABC):
    """
    Abstract base class for source data loaders.
    """

    def __init_subclass__(cls):
        PSRLCFG.CLASS_REGISTRY["DATA_LOADER"][cls.__name__] = cls

    @abstractmethod
    def get_l1(self, source_file: pd.Series, **kwargs) -> Optional[Level1bData]:
        """
        Abstract method to load source data into Level 1 data object from a given file path.
        """
        pass


class FileDiscovery(ABC):
    """
    Abstract base class for file discovery.
    """

    def __init_subclass__(cls):
        PSRLCFG.CLASS_REGISTRY["FILE_DISCOVERY"][cls.__name__] = cls

    @abstractmethod
    def query_period(self, period: DatePeriod) -> Optional[List[Path]]:
        """
        Abstract method to load source data into Level 1 data object from a given file path.
        """
        pass
