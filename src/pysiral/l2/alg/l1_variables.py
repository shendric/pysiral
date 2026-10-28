# -*- coding: utf-8 -*-
"""
This module contains the Level-2 algorithms that transfer Level-1 information
to the Level-2 data object.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = []

from loguru import logger
from itertools import product
from typing import List

import numpy as np

from pysiral.l2 import L2Algorithm


class L1BL2TransferVariables(L2Algorithm):
    """
    Level-2 Processor step class to transfer variables from the
    l1b to the l2 data object
    """

    def __init__(self, cfg, **kwargs):
        """
        Initiliaze the class
        :param arg:
        :param kwargs:
        """
        self.cfg = cfg

    def apply(self, l1: "Level1bData", l2: "Level2Data") -> np.ndarray:
        """
        Mandatory method of Level-2 processor
        :param l1: Level-1 data object
        :param l2: Level-2 data object
        :return: error_status
        """

        # Get the error mandatory
        error_status = self.get_clean_error_status(l2.n_records)

        logger.info("- Transfer L1P variables to L2")
        for data_group, varlist in list(self.cfg.options.items()):

            # Get and loop over variables per data group
            l1p_variables = varlist.items()
            for var_name, vardef in list(l1p_variables):

                # Get variable via standard getter method
                # NOTE: Will return None if not found -> create an empty array
                #       -> this will be noted in the error status flag
                var = l1.get_parameter_by_name(data_group, var_name)
                if var is None:
                    logger.warning(f"Cannot find variable {data_group}.{var_name} in l1p")
                    var = np.full(l2.n_records, np.nan)
                    error_status = np.logical_not(error_status)

                # Add variable to l2 object as auxiliary variable
                aux_id, aux_name = vardef
                l2.set_auxiliary_parameter(aux_id, aux_name, var, None)

        return error_status

    @property
    def l2_input_vars(self) -> List[str]:
        return []

    @property
    def l2_output_vars(self) -> List[str]:
        output_vars = []
        for data_group, varlist in list(self.cfg.options.items()):
            l1p_variables = varlist.items()
            output_vars.extend(vardef[0] for var_name, vardef in list(l1p_variables))
        return output_vars

    @property
    def error_bit(self) -> int:
        return self.error_flag_bit_dict["l2"]


class TransferWaveform2L2(L2Algorithm):
    """
    This class adds the waveform and associated parameters (range, valid_flag)
    to the l2 data container

        - module: l2.procsteps
          pyclass: TransferWaveform2L2
          options: {}
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def apply(self, l1: "Level1bData", l2: "Level2Data") -> np.ndarray:
        """
        API method for Level2ProcessorStep subclasses. Computes and add rolling statistics
        based on the processor item options in the Level-2 processor definition file.

        :param l1: Level-1 data object
        :param l2: Level-2 data object

        :raises None:

        :return: error_status
        """

        error_status = self.get_clean_error_status(l2.n_records)

        l2.set_auxiliary_parameter("wfmvf", "waveform_valid_flag", l1.waveform.is_valid)

        # Waveform and waveform model
        # Register results as auxiliary data variable
        num_range_gates = l1.waveform.power.shape[1]
        dims = {"new_dims": (("range_gates", num_range_gates),),
                "dimensions": ("time", "range_gates"),
                "add_dims": (("range_gates", np.arange(num_range_gates)),)}
        l2.set_multidim_auxiliary_parameter("wfmp", "waveform_power", l1.waveform.power, dims, update=True)
        l2.set_multidim_auxiliary_parameter("wfmr", "waveform_range", l1.waveform.range, dims, update=True)

        return error_status

    @property
    def statistics_combinations(self):
        return product(self.cfg.options.input_parameters, self.cfg.options.statistics)

    @property
    def l2_dependencies(self):
        return []

    @property
    def l2_output_vars(self):
        return ["waveform_power", "waveform_range", "waveform_valid_flag"]

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["other"]