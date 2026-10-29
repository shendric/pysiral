# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = ["L1PLeadingEdgePeakiness", "LeadingEdgePeakinessConfig"]

import bottleneck as bn
import numpy as np
from loguru import logger
from pydantic import BaseModel

from pysiral.l1 import L1Algorithm
from pysiral.l1.l1_data import Level1bData

class LeadingEdgePeakinessConfig(BaseModel):
    """
    Configuration class for the leading edge peakiness algorithm.
    """
    window_size: int = 5


class L1PLeadingEdgePeakiness(L1Algorithm):

    def __init__(self, **cfg):
        self.cfg = LeadingEdgePeakinessConfig(**cfg)

    def apply(self, l1: Level1bData):
        """
        Mandatory class of a L1 preproceessor item. Computes the leading edge peakiness

        :param l1:
        :return:
        """

        # Init the classifier (leading edge peakiness
        lep = np.full(l1.info.n_records, np.nan)

        # Get the waveform power
        wfm = l1.waveform.power

        # Get the first maximum index
        fmi = l1.classifier.get_parameter("first_maximum_index", raise_on_error=False)
        if fmi is None:
            logger.error("Classifier `first_maximum_index` not available -> skipping leading edge peakiness")
            l1.classifier.add(lep, "leading_edge_peakiness")
            return

        # Loop over all waveforms
        for i in np.arange(wfm.shape[0]):
            if fmi[i] < 0:
                continue
            lep[i] = self.leading_edge_peakiness(wfm[i, :], fmi[i], self.cfg.window_size)

        # Convert inf to nan
        lep[np.isinf(lep)] = np.nan

        # Add to classifier
        l1.classifier.add(lep, "leading_edge_peakiness")

    @staticmethod
    def leading_edge_peakiness(wfm: np.ndarray, fmi: int, window: int) -> float:
        """
        Compute the leading edge peakiness
        :param wfm: Waveform power
        :param fmi: first maximum index
        :param window: the number or leading range bins to the first maximum for the
            peakiness computation
        :return:
        """

        # Get the waveform subset prior to first maximum
        i0 = fmi - window
        i0 = max(i0, 0)
        return wfm[fmi] / bn.nanmean(wfm[i0:fmi]) * (fmi - i0)

