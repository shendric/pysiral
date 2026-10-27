# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from abc import ABC, abstractmethod
from pysiral import PSRLCFG
from pysiral.l1 import Level1bData
from pysiral.l2 import Level2Data


class L2Algorithm(ABC):
    """
    Abstract base class for Level-2 PreProcessor Algorithms.
    """

    def __init_subclass__(cls):
        PSRLCFG.CLASS_REGISTRY["L2_ALGORITHM"][cls.__name__] = cls

    @abstractmethod
    def apply(self, l1: "Level1bData", l2: "Level2Data") -> None:
        """
        Mandatory class for Level-1 processing item. Needs to be overwritten
        by the inheriting class.
        :param l1:

        :return:
        """
        pass
