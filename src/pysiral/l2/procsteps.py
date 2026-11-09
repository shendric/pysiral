# -*- coding: utf-8 -*-
"""
@author: Stefan Hendricks
"""

from itertools import product

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import List, Type
from loguru import logger
from scipy.interpolate import interp1d

from pysiral import PSRLCFG
from pysiral.core.legacy_classes import DefaultLoggingClass
from pysiral.l1.l1_data import Level1bData
from pysiral.l2 import L2Algorithm
from pysiral.l2.l2data import Level2Data


class Level2ProcessorStepOrder(DefaultLoggingClass):
    """
    A container providing the ordered list of processing steps
    as initialized classes for each trajectory
    """

    def __init__(self, cfg: List["L2ProcAlgorithmDef"]):
        """
        Initialize this class
        :param cfg: the procsteps tag from the Level-2 processor definitions file
        """
        super(Level2ProcessorStepOrder, self).__init__(self.__class__.__name__)

        # Properties
        self.cfg = cfg

        # A list of the class object (not initialized!)
        self._classes = self.get_procstep_classes()

    def get_procstep_classes(self) -> List[Type]:
        """
        Retrieves the required classes from the processor definition files and stores them in a list
        without initializing them. This way a freshly initialized version can be supplied to each
        l2 data object without risk of interference of class properties
        :return:
        """
        l2_classes = []
        for procstep_def in self.cfg:
            obj = PSRLCFG.get_class_from_registry("L2_ALGORITHM", procstep_def.class_name)
            logger.info(f"Added L2 processor step: {procstep_def.class_name}")
            l2_classes.append(obj)
        return l2_classes

    def get_algorithm_error_flag_bit(self, procstep_module):
        """
        Return the corresponding bit (0-15) for the procstep module
        :param procstep_module: str
        :return:
        """
        pass

    def validate(self) -> None:
        """
        Checkout the difference processing steps and validate input/output variables in
        the order of the steps
        :return:
        """
        # Get the initialized classes and check if it has the required methods
        classes = self.class_instances
        for obj in classes:
            # Check if the class is a subclass of L2Algorithm
            assert issubclass(obj.__class__, L2Algorithm), f"Not a valid Level-2 algorithm class {obj.__class__}"

    @property
    def class_list(self):
        """
        Return a list of not initialized classes (class instances with the options from the level-2
        :return:
        """
        return list(self._classes)

    @property
    def class_instances(self):
        """
        Return a list of initialized classes (class instances with the options from the level-2 processor
        definition file passed to the class instance)
        :return:
        """
        return [pyclass(opt) for pyclass, opt in zip(self.class_list, self.cfg)]
