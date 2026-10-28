# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from loguru import logger
from scipy.interpolate import interp1d
from typing import List

import numpy as np
import numpy.typing as npt

from pysiral.l2 import L2Algorithm


class L2ApplyRangeCorrections(L2Algorithm):
    """
    Level-2 processing step class for applying geophysical range corrections to the elevations
    """

    def __init__(self, cfg):
        """
        Init the class
        :param args:
        :param kwargs:
        """
        self.cfg = cfg

    def apply(self, l1, l2):
        """
        Mandatory method of Level-2 processor
        :param l1:
        :param l2:
        :return: error_status
        """

        # Get the error mandatory
        error_status = self.get_clean_error_status(l2.n_records)

        # Keep the total range correction
        total_range_correction = np.full(l2.n_records, 0.0)

        preserve_nan = self.cfg.options.get("preserve_nan", False)

        # Apply the range corrections (content of l1 data _package)
        # to the l2 elevation (output of retracker) data
        for correction_name in self.cfg.options.corrections:

            # Get the range correction field
            range_delta = l2.get_parameter_by_name(correction_name)

            # Check if data is in l1p
            # -> skip with warning if not
            if range_delta is None:
                error_status[:] = True
                logger.error(f"- Cannot find range correction: {correction_name} - skipping")
                continue

            # Check if NaN's, set values to zero and provide warning
            if np.isnan(range_delta).all():
                logger.warning(f"{correction_name}: - only NaNs [all set to 0.0]")
                range_delta = np.zeros(l2.n_records)
                l2.update_parameter(correction_name, range_delta)
                error_status[:] = True
            elif np.isnan(range_delta).any():
                if preserve_nan:
                    logger.warning(f"{correction_name}: - NaNs encountered [preserved]")
                else:
                    logger.warning(f"{correction_name}: - NaNs encountered [interpolated]")
                    range_delta = self.fill_nan_gaps(range_delta)
                    l2.update_parameter(correction_name, range_delta)
                nan_idxs = np.where(np.isnan(range_delta))[0]
                error_status[nan_idxs] = True

            # Apply the range correction
            #
            # NOTES:
            #
            #    1. It is assumed at this point that the uncertainty of the range correction is
            #       negligible compared to the remaining uncertainty components, respectively
            #       is already included in the general range uncertainty budget
            #
            #    2. the range correction variable `range_delta` is subtracted from the elevation
            #       as it has to be added to range:
            #
            #           elevation = altitude - (range + range_correction)
            #       ->  elevation = altitude - range - range_correction
            #                       |              |
            #                       ----------------
            #                 elevation after retracking
            #
            for target_variable in self.target_variables:
                var = l2.get_parameter_by_name(target_variable)
                var[:] = var[:] - range_delta
                l2.set_parameter(target_variable, var[:], var.uncertainty[:])

            # store the range correction in the total range correction array
            total_range_correction += range_delta

        # Add the total range correction to the l2 auxiliary variables
        l2.set_auxiliary_parameter("rctotal", "total_range_correction", total_range_correction)

        return error_status

    @staticmethod
    def fill_nan_gaps(
            range_correction: npt.NDArray,
            extrapolate: bool = True,
    ) -> npt.NDArray:
        """
        Fills gaps in range correction. If the range correction is all-NaN,
        just return zeros

        :param range_correction:
        :param extrapolate:
        :return:
        """

        # Get list of finite range correction values
        x = np.arange(range_correction.shape[0])
        valid_idx = np.where(np.isfinite(range_correction))[0]

        # Return zero range correction (no change) for all-NaN parameter
        if valid_idx.size == 0:
            return np.zeros(range_correction.shape)

        # Interpolate and, if specified extrapolate range correction
        fill_value = "extrapolate" if extrapolate else np.nan
        interp_func = interp1d(
            x[valid_idx],
            range_correction[valid_idx],
            bounds_error=False,
            fill_value=fill_value
        )
        return interp_func(x)

    @property
    def target_variables(self):
        """
        Return target variables from config file or default value
        :return:
        """
        return self.cfg.options.get("target_variables", ["elevation"])

    @property
    def l2_dependencies(self) -> List[str]:
        return self.target_variables

    @property
    def l2_output_vars(self):
        return ["rctotal"]

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["range_correction"]