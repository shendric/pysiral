# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


from typing import List, Dict

import bottleneck as bn
import numpy as np
import numpy.typing as npt
from loguru import logger

from pydantic import BaseModel

from pysiral.core.flags import RadarModes
from pysiral.l1 import L1Algorithm
from pysiral.l1.l1_data import Level1bData




class LateTail2PeakPowerConfig(BaseModel):
    """
    Configuration class for the late tail to peak power algorithm.
    """
    late_tail_window_idx: Dict[str, List[int]]


class L1PLateTail2PeakPower(L1Algorithm):

    def __init__(self, **cfg) -> None:
        self.cfg = LateTail2PeakPowerConfig(**cfg)

    def apply(self, l1: Level1bData) -> None:
        """
        Compute late tail to peakiness and add to l1 data object
        :param l1:

        :return:
        """

        # Init the classifier
        ltpp = np.full(l1.info.n_records, np.nan)


        # Loop over all waveforms and compute ltpp
        wfm = l1.waveform.power
        radar_mode = l1.waveform.radar_mode
        invalid_radar_mode_encountered = 0

        for i in np.arange(wfm.shape[0]):
            late_tail_window_idx = self.cfg.late_tail_window_idx.get(
                RadarModes.get_name(radar_mode[i]), None
            )
            if not late_tail_window_idx:
                ltpp[i] = np.nan
                invalid_radar_mode_encountered += 1
                continue
            ltpp[i] = self.late_tail_to_peak_power(wfm[i, :], late_tail_window_idx)

        if invalid_radar_mode_encountered > 0:
            logger.warning(
                f"Could not compute ltpp for {invalid_radar_mode_encountered} waveforms [invalid radar mode]"
            )
        l1.classifier.add(ltpp, "late_tail_to_peak_power")

    @staticmethod
    def late_tail_to_peak_power(
            wfm: npt.NDArray,
            late_tail_window_idxs: List[int]
    ) -> float:
        """

        :param wfm:
        :param late_tail_window_idxs:
        :return:
        """

        # Get Properties
        wfm_max_power = bn.nanmax(wfm)
        try:
            wfm_max_power_index = bn.nanargmax(wfm)
        except ValueError:  # E.g. all-nan waveform
            return np.nan

        late_tail_window = np.arange(
            wfm_max_power_index + late_tail_window_idxs[0],
            wfm_max_power_index + late_tail_window_idxs[1] + 1
        )

        # Do not compute ltpp if late tail window is incomplete
        if (late_tail_window >= wfm.shape[0]).any():
            return np.nan

        try:
            return bn.nanmean(wfm[late_tail_window])/wfm_max_power
        except ZeroDivisionError:
            return np.nan
