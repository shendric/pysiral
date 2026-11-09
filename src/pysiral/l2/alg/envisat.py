# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from pysiral.l2 import L2Algorithm

class EnvisatBackscatterDriftCorrection(L2Algorithm):
    """
    Very specific filter to correct backscatter drift. The filter
    applies a monthly linear correction based on the drift factor and
    base period.

    Notes:

        1. This class is very specifically designed for correcting
           the sigma0 value of the degrading Envisat RA-2 antenna
    """

    def __init__(self, cfg):
        super(EnvisatBackscatterDriftCorrection, self).__init__()
        self.cfg = cfg

    def apply(self, l1, l2):
        """
        Apply a backscatter correction
        :param l1:
        :param l2:
        :return:
        """

        # Get the default error status
        error_status = self.get_clean_error_status(l2.n_records)

        # Get the backscatter value
        datagroup = self.cfg.options.l1b_data_group
        name = self.cfg.options.l1b_parameter_name
        value = l1.get_parameter_by_name(datagroup, name)

        # Compute the drift correction
        year_base, month_base = self.cfg.options.backscatter_base_period
        year, month = l1.info.start_time.year, l1.info.start_time.month
        time_shift_factor = (year_base - year) * 12 + (month_base - month)
        sigma0_drift_factor = self.cfg.options.backscatter_drift_factor
        sigma0_drift = time_shift_factor * sigma0_drift_factor

        # Apply and update the backscatter by changing the l1 data object in-place
        value += sigma0_drift
        l1.set_parameter_by_name(datagroup, name, value)

        # Done
        return error_status

    @property
    def l2_input_vars(self):
        return []

    @property
    def l2_output_vars(self):
        return []

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["other"]