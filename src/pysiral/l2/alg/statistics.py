# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from itertools import product
from loguru import logger

import numpy as np
import pandas as pd

from pysiral.l2 import L2Algorithm


class ParameterRollingStatistics(L2Algorithm):
    """
    This class add rolling statistics (mean, sdev) using the pandas rolling
    framework to the Level-2 data object.

        -   module: l2.procsteps
            pyclass: ParameterRollingStatistics
            options:
                window_size_m: <maximum size of the marginal ice zone>
                statistics: [ <list of statistics: "mean" | "sdev"]
                input_parameters:
                    - <l2_parameter-01>
                    - <l2_parameter-02>
                    - ...

    For each parameter & statistics combination a new parameter will be added
    with the naming convention:

                <source_parameter_name>_rolling_<statistics_id>

        e.g.

        -   module: l2.procsteps
            pyclass: ParameterRollingStatistics
            options:
                window_size_m: 25000.
                statistics: ["sdev"]
                input_parameters:
                    - pulse_peakiness

        will add the parameter `pulse_peakiness_rolling_sdev` to the list of l2 (auxiliary)
        data parameters.
    """

    def __init__(self, cfg):
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

        # The filter windows size needs to be an odd integer
        window_size_float = self.cfg.options.get("window_size_m") / l2.footprint_spacing
        window_size = None if np.isnan(window_size_float) else int(int(window_size_float) // 2 * 2 + 1)

        rolling_kwargs = dict(window=window_size, center=True, min_periods=1)
        for parameter_name, statistics_id in self.statistics_combinations:

            aux_id = f"{l2.full_variable_catalog[parameter_name]}rl{statistics_id}"
            aux_name = f"{parameter_name}_rolling_{statistics_id}"

            if window_size is None:
                error_status[:] = True
                l2.set_auxiliary_parameter(aux_id, aux_name, np.full(l2.n_records, np.nan), None)
                continue

            # The rolling statistics are computed using the pandas rolling framework
            ts = pd.Series(l2.get_parameter_by_name(parameter_name)[:])
            if statistics_id == "sdev":
                ts_rolled = ts.rolling(**rolling_kwargs).std()
            elif statistics_id == "mean":
                ts_rolled = ts.rolling(**rolling_kwargs).mean()
            else:
                logger.error(f"{self.__class__.__name__}: Encountered invalid statistic name: {statistics_id}")
                error_status[:] = True
                ts_rolled = pd.Series(np.full(l2.n_records, np.nan))

            # Output is added as an auxiliary parameter
            l2.set_auxiliary_parameter(aux_id, aux_name, ts_rolled.values, None)

        return error_status

    @property
    def statistics_combinations(self):
        return product(self.cfg.options.input_parameters, self.cfg.options.statistics)

    @property
    def l2_dependencies(self):
        return [self.cfg.options.input_parameters]

    @property
    def l2_output_vars(self):
        return [f"{varname}_rolling_{statname}" for (varname, statname) in self.statistics_combinations]

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["other"]