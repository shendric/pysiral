# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from pysiral.l2 import L2Algorithm


class CS2InstrumentModeflag(L2Algorithm):
    """
    A class creating an instrument_mode flag from radar_mode:

        instrument_mode = radar_mode + 1
    """

    def __init__(self, cfg, **kwargs):
        """
        Init the class
        :param cfg:
        :param kwargs:
        """
        super(CS2InstrumentModeflag, self).__init__()
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
