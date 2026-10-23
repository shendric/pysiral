# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


from dataclasses import field
from loguru import logger
from pydantic import BaseModel
from typing import Dict

import numpy as np


from pysiral.l1 import L1Algorithm
from pysiral.retracker.tfmra import cTFMRA


class LeadingEdgeQualityConfig(BaseModel):
    """
    Configuration class for the L1PLeadingEdgeQuality pre-processor item
    """
    leading_edge_lookup_window: Dict[str, int] = field(default_factory=dict)
    first_maximum_normalized_power_threshold: Dict[str, float] = field(default_factory=dict)
    minimum_valid_first_maximum_index: Dict[str, int] = field(default_factory=dict)


class L1PLeadingEdgeQuality(L1Algorithm):
    """
    Class to compute a leading edge width quality indicator
    Requires `first_maximum_index` classifier parameter
    """

    def __init__(self, **cfg):
        self.cfg = LeadingEdgeQualityConfig(**cfg)

    def apply(self, l1):
        """
        Adds a quality indicator for the leading edge

        :param l1: l1bdata.Level1bData instance

        :return: None
        """

        # Get the waveform power
        wfm_power = l1.waveform.power

        # Create output parameters
        leq = np.full(l1.info.n_records, np.nan)              # leading edge quality
        fmi = np.full(l1.info.n_records, -1, dtype=int)       # first maximum index
        fmp = np.full(l1.info.n_records, np.nan)              # first maximum power fraction (to peak power)

        # --- Get the required options ---

        # The window for the quality computation depends on the radar mode
        window = self.cfg.leading_edge_lookup_window.get(l1.radar_modes, None)
        if window is None:
            logger.error(f"leading_edge_lookup_window not defined for radar mode: {l1.radar_modes}")
            return

        # The power threshold depends on the radar mode
        power_threshold = self.cfg.first_maximum_normalized_power_threshold.get(l1.radar_modes, None)
        if power_threshold is None:
            logger.error(f"first_maximum_normalized_power_threshold not defined for radar mode: {l1.radar_modes}")
            return

        # Normalized power threshold for identifications of the first maximum

        # The power threshold depends on the radar mode
        fmi_min = self.cfg.minimum_valid_first_maximum_index.get(l1.radar_modes, None)
        if fmi_min is None:
            logger.error(f"minimum_valid_first_maximum_index not defined for radar mode: {l1.radar_modes}")
            return

        # Loop over all waveforms
        for i in np.arange(l1.info.n_records):

            # Prepare the waveform data
            wfm = wfm_power[i, :].astype(float)    # Get the subset and ensure data type float
            wfm /= np.nanmax(wfm)                  # Normalize (requires float)

            # Get the first maximum index
            fmi_idx = cTFMRA.get_first_maximum_index(wfm, power_threshold, first_valid_idx=fmi_min)
            if fmi_idx == -1 or fmi_idx < fmi_min:
                continue
            fmi[i] = fmi_idx

            # Save the power values
            fmp[i] = wfm[fmi[i]]

            # Get the search range
            i0, i1 = fmi[i]-window, fmi[i]+1
            i0 = max(i0, 1)
            power_diff = wfm[i0:i1]-wfm[i0-1:i1-1]
            positive_power_diff = power_diff[power_diff > 0]
            total_power_raise = np.sum(positive_power_diff) + wfm[i0]

            # Leading edge quality indicator
            leq[i] = total_power_raise / fmp[i]

        # Add the classifier to the l1 object
        l1.classifier.add(leq, "leading_edge_quality")
        l1.classifier.add(fmi, "first_maximum_index")
        l1.classifier.add(fmp, "first_maximum_power")