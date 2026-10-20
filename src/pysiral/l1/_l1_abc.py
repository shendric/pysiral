# -*- coding: utf-8 -*-

"""
This module contains Abstract Baseclasses used for the Level-1 Pre-Processor and its
compontens (e.g. algorithms).
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from abc import ABC, abstractmethod
from pysiral import PSRLCFG


class SourceDataLoader(ABC):
    """
    Abstract base class for Level-1 PreProcessor Algorithms.
    """

    def __init_subclass__(cls):
        PSRLCFG.CLASS_REGISTRY["L1_ALGORITHM"][cls.__name__] = cls

    @abstractmethod
    def apply(self, l1: "Level1bData") -> None:
        """
        Mandatory class for Level-1 processing item. Needs to be overwritten
        by the inheriting class.
        :param l1:

        :return:
        """
        pass
