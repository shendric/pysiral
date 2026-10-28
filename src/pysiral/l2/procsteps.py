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


class Level2ProcessorStep(DefaultLoggingClass):
    """
    Parent class for any Level-2 processor step class, which may be distributed over the
    different pysiral modules.

    This class also serves as a template for all subclasses. Mandatory methods and properties
    in this class which raise a NotImplementedException must be overwritten by the subclass
    """

    def __init__(self, cfg):
        """
        Init the
        :param cfg:
        """
        super(Level2ProcessorStep, self).__init__(self.__class__.__name__)

        # -- Properties --

        # Class configuration
        self.cfg = cfg

        # Log messages
        self.msgs = []

        # Error flag dict {code: component}
        self.error_flag_bit_dict = {
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
            "other": 15}

    def execute(self, l1b: Level1bData, l2: Level2Data) -> None:
        """
        The main entry point for the
        :param l1b:
        :param l2:
        :return:
        """

        # Execute the method of the subclass. The class needs to
        error_status = self.execute_procstep(l1b, l2)

        # Check if the error status is correctly implemented
        if error_status is None:
            logger.warning(f"Class {self.classname} does not provide error status")
            error_status = self.get_clean_error_status(l2.n_records)

        # Update the status flag
        self.update_error_flag(l2, error_status)

    def update_error_flag(self, l2, error_status):
        """
        Add the error_flag of the processing step to the

        :param: l2: The Level-2 data container
        :param: error_status: An array with the shape of l2.records containing the error flag
            (False: nominal, True: error)
        :return:
        """
        # Update the algorithm error flag in the Level-2 data object
        # NOTE: The bit for which the algorithm error flag is applied depends on the
        #       mandatory value set in each child class of Level2ProcessorStep
        flag_value = 2**self.error_bit
        flag = error_status.astype(int)*flag_value
        l2.flag = l2.flag + flag

    def execute_procstep(self, l1b, l2):
        raise NotImplementedError(
            f"This method needs to implemented in {self.classname}"
        )

    @staticmethod
    def get_clean_error_status(shape):
        """
        Return an empty (all nominal error status array)
        :param shape:
        :return:
        """
        return np.full(shape, False)

    @property
    def classname(self):
        return self.__class__.__name__

    @property
    def error_bit(self):
        raise NotImplementedError(
            f"This method needs to implemented in {self.classname}"
        )

    @property
    def l2_input_vars(self):
        raise NotImplementedError(
            f"This method needs to implemented in {self.classname}"
        )

    @property
    def l2_output_vars(self):
        raise NotImplementedError(
            f"This method needs to implemented in {self.classname}"
        )


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





class CS2InstrumentModeflag(Level2ProcessorStep):
    """
    A class creating an instrument_mode flag from radar_mode:

        instrument_mode = radar_mode + 1
    """

    def __init__(self, *args, **kwargs):
        """
        Init the class
        :param args:
        :param kwargs:
        """
        super(CS2InstrumentModeflag, self).__init__(*args, **kwargs)

    def execute_procstep(self, l1b, l2):
        """
        Mandatory method of Level-2 processor
        :param l1b:
        :param l2:
        :return: error_status
        """

        # Get the error mandatory
        error_status = self.get_clean_error_status(l2.n_records)

        # Instrument mode is a variant of radar mode where the flag starts at 1 and not 0
        instrument_mode = l2.radar_mode + 1

        # Add the total range correction to the l2 auxiliary variables
        l2.set_auxiliary_parameter("imode", "instrument_mode", instrument_mode)

        return error_status

    @property
    def l2_dependencies(self):
        return ["radar_mode"]

    @property
    def l2_output_vars(self):
        return ["instrument_mode"]

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["other"]






