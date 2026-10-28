# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import numpy as np

from pysiral.l2 import L2Algorithm


class ERSPulseDeblurring(L2Algorithm):
    """
    A processing step applying the pulse deblurring correction of
    Peacock 1998 to retracked ranges.

    NOTE: Should only be used for ERS-1/2 data
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, l1, l2):
        """
        Compute the pulse deblurring correction

            Hcor = h + eps / 5. (eps < 0)

        based on the classifier data transferred from the l1p (epss: epsilon in seconds).
        :param l1:
        :param l2:
        :return: error_status_flag
        """

        # Get a clean error status
        error_status = self.get_clean_error_status(l2.n_records)

        # Compute epsilon in meter (eps_m = eps_sec * c / 2.)
        eps = l2.epss * 0.5 * 299792458.

        # Compute and apply pulse deblurring correction
        slope = self.constant_slope
        pulse_deblurring_correction = np.array(eps < 0.).astype(float) * eps / slope
        for target_variable in self.target_variables:
            var = l2.get_parameter_by_name(target_variable)
            var[:] = var[:] + pulse_deblurring_correction
            l2.set_parameter(target_variable, var[:], var.uncertainty[:])

        # Add pulse deblurring correction to level-2 auxiliary data
        l2.set_auxiliary_parameter("pdbc", "pulse_deblurring_correction", pulse_deblurring_correction)

        # Return clean error status (for now)
        return error_status

    @property
    def l2_dependencies(self):
        return ["elev", "epss"]

    @property
    def l2_output_vars(self):
        return ["pdbc"]

    @property
    def target_variables(self):
        return self.cfg.options.get("target_variables", ["elevation"])

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["retracker"]

    @property
    def constant_slope(self):
        # return slope term depending on ERS mission
        # defaults to 5.0, the ERS-2 value
        return self.cfg.options.get("slope", 5.0)