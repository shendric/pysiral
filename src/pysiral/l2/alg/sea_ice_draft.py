# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = ["L2SeaIceDraft"]

import numpy as np

from pysiral.l2 import L2Algorithm


class L2SeaIceDraft(L2Algorithm):
    """
    A Level-2 processor step class for computing sea ice draft and its uncertainty
    """

    def __init__(self,  cfg):
        self.cfg = cfg

    def apply(self, l1, l2):
        """
        API class for the Level-2 processor. Functionality is to compute sea ice draft and its uncertainty
        :param l1: Level-1 data instance
        :param l2: A Level-2 data instance
        :return: None, Level-2 object is change in place
        """

        # Get error status
        error_status = self.get_clean_error_status(l2.n_records)

        # Compute sea-ice draft
        sea_ice_draft = l2.sit - l2.frb
        sea_ice_draft_uncertainty = np.sqrt(l2.sit.uncertainty**2. + l2.frb.uncertainty**2.)

        # Add to l2
        l2.set_auxiliary_parameter("sid", "sea_ice_draft", sea_ice_draft, uncertainty=sea_ice_draft_uncertainty)

        return error_status

    @property
    def l2_dependencies(self):
        return ["frb", "sit"]

    @property
    def l2_output_vars(self):
        return ["sid"]

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["sit"]