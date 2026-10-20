# -*- coding: utf-8 -*-

"""
This module contains Level-1 Pre-Processor algorithms specifically for the FDR4ALT dataset.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


from pysiral.l1 import L1Algorithm


class FillInRangeCorrectionsFromTSIP(L1Algorithm):

    def __init__(self, **kwargs) -> None:
        """
        Level-1 Pre-Processor algorithm to fill in missing range corrections from TSIP data.

        :param kwargs: Configuration parameters for the algorithm
        """
        pass

    def apply(self, l1):
        """
        Apply the algorithm to the Level-1 data.

        :param l1: Level-1 data object
        """
        breakpoint()
