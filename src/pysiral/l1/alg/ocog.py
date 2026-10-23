# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from typing import List
import bottleneck as bn
import numpy as np

from pysiral.l1.l1_data import Level1bData
from pysiral.l1.alg import L1PProcItem



class OCOGParameter(object):
    """
    Calculate OCOG Parameters (Amplitude, Width) for CryoSat-2 waveform
    counts.
    Algorithm Source: retrack_ocog.pro from CS2AWI lib
    """

    def __init__(self, wfm_counts):
        self._n = np.shape(wfm_counts)[0]
        self._amplitude = np.ndarray(shape=[self._n], dtype=np.float32)
        self._width = np.ndarray(shape=[self._n], dtype=np.float32)
        self._calc_parameters(wfm_counts)

    def _calc_parameters(self, wfm_counts):
        for i in np.arange(self._n):
            y = wfm_counts[i, :].flatten().astype(np.float64)
            y -= bn.nanmean(y[:11])  # Remove Noise
            y[np.where(y < 0.0)[0]] = 0.0  # Set negative counts to zero
            y2 = y ** 2.0
            self._amplitude[i] = np.sqrt((y2 ** 2.0).sum() / y2.sum())
            self._width[i] = ((y2.sum()) ** 2.0) / (y2 ** 2.0).sum()

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def width(self):
        return self._width


class L1POCOG(L1PProcItem):
    """
    Computes OCOG (Offset Centre of Gravity) width and amplitude
    as a Level-1 pre-processor item
    """

    def __init__(self, **cfg):
        super(L1POCOG, self).__init__(**cfg)

    def apply(self, l1: Level1bData) -> None:
        """
        Compute OCOG Width and Amplitude for each waveform
        and add to classifier data group

        :param l1: Level-1 data object
        """
        ocog = OCOGParameter(l1.waveform.power)
        l1.classifier.add(ocog.width, "ocog_width")
        l1.classifier.add(ocog.amplitude, "ocog_amplitude")

    @property
    def required_options(self) -> List[str]:
        return []


