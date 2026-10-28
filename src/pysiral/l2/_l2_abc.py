# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from abc import ABC, abstractmethod, abstractproperty
from typing import List, Dict

import numpy as np

from pysiral import PSRLCFG


class L2Algorithm(ABC):
    """
    Abstract base class for Level-2 PreProcessor Algorithms.
    """

    def __init_subclass__(cls):
        """
        Register the subclass in the PSRLCFG.CLASS_REGISTRY under the "L2_ALGORITHM" key.
        This allows for dynamic discovery and instantiation of Level-2 algorithms.

        :param cls: The subclass being registered.
        """
        PSRLCFG.CLASS_REGISTRY["L2_ALGORITHM"][cls.__name__] = cls

    @abstractmethod
    def apply(self, l1: "Level1bData", l2: "Level2Data") -> np.ndarray:
        """
        [Mandatory class method] Apply the algorithm to Level-2 data object,
        based on the Level-1 & Level-2 data

        :param l1: Level-1 data object
        :param l2: Level-2 data object

        :return: Error flag bit array (np.ndarray) indicating which records failed the algorithm
        """
        pass

    @abstractproperty
    def l2_dependencies(self) -> List[str]:
        """
        [Mandatory class method] List the Level-2 variables that are required for this algorithm to run.

        :return: List of variable names that are required for this algorithm to run
        """
        pass

    @abstractproperty
    def l2_output_vars(self) -> List[str]:
        """
        [Mandatory class method] List the Level-2 variables that are produced by this algorithm.

        :return: List of variable names that are produced by this algorithm
        """
        pass

    @abstractproperty
    def error_bit(self) -> int:
        """
        [Mandatory class method] Return the error bit that is associated with this algorithm.
        This bit will be set in the Level-2 data object if the algorithm fails for a given record.

        :return: Error bit (int) associated with this algorithm
        """
        pass

    @staticmethod
    def get_clean_error_status(shape) -> np.ndarray:
        """
        Return an empty (all nominal error status array)
        :param shape:
        :return: np.ndarray
        """
        return np.full(shape, False)

    @property
    def errpr_flag_bit_dict(self) -> Dict[str, int]:
        """
        Return the error flag bit dictionary that maps error types to their corresponding bit positions.
        :return: Dictionary mapping error types to bit positions
        """
        return {
            "l1b": 0,
            "l2": 1,
            "auxdata": 2,
            "surface_type": 3,
            "retracker": 4,
            "range_correction": 5,
            "sla": 6,
            "frb": 7,
            "sit": 8,
            "filter": 9,
            "other": 15
        }

