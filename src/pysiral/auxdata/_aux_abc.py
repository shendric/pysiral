# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from abc import ABC, abstractmethod
from pysiral import PSRLCFG
# from pysiral.l1.l1_data import L1bdataNCFile
# from pysiral.l2.l2data import Level2Data


class AuxDataLoader(ABC):
    """
    Abstract base class for Level-2 PreProcessor Algorithms.
    """

    def __init_subclass__(cls):
        PSRLCFG.CLASS_REGISTRY["AUX_DATA_LOADER"][cls.__name__] = cls

    @abstractmethod
    def receive_l1p_input(self, l1p: 'L1bdataNCFile') -> None:
        """
        Mandatory class for Level-1 processing item. Needs to be overwritten
        by the inheriting class.
        :param l1:

        :return:
        """
        pass

    @abstractmethod
    def get_l2_track_vars(self, l2: 'Level2Data') -> None:
        """
        [Mandatory class method] Add the model prediction for the tfmra retracker threshold to the
        Level-2 data object. The model evaluation depends solely on waveform power.

        :param l2: `Level2Data` container

        :return: None
        """
        pass