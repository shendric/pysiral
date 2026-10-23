# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from typing import Union

import bottleneck as bn
import numpy as np
import numpy.typing as npt

from pydantic import BaseModel

from pysiral.l1 import L1Algorithm
from pysiral.l1.l1_data import Level1bData


class WaveformPeakinessConfig(BaseModel):
    """
    Configuration class for the L1PWaveformPeakiness pre-processor item
    """
    skip_first_range_bins: int = 0
    norm_is_range_bin: bool = True


class L1PWaveformPeakiness(L1Algorithm):
    """
    A L1P pre-processor item class for computing pulse peakiness """

    def __init__(self, **kwargs):
        self.cfg = WaveformPeakinessConfig(**kwargs)

    def apply(self, l1: Level1bData) -> None:
        """
        Computes pulse peakiness and adds parameter to classifier data group.

        NOTE: The classifier parameter name depends on the `norm_is_range_bin keyword:

            norm_is_range_bin = True -> parameter name: 'peakiness'
            norm_is_range_bin = False -> parameter name: 'peakiness_normed'

        :param l1: l1bdata.Level1bData instance

        :raises None:

        :return: None
        """
        waveforms = l1.waveform.power
        pulse_peakiness = self.compute_for_waveforms(waveforms)
        parameter_target_name = "peakiness" if self.cfg.norm_is_range_bin else "peakiness_normed"
        l1.classifier.add(pulse_peakiness, parameter_target_name)

    def compute_for_waveforms(self, waveforms: npt.NDArray) -> npt.NDArray:
        """
        Compute pulse peakiness for a waveform array

        :param waveforms:

        :return: pulse peakiness array
        """

        # Get the waveform
        n_records, n_range_bins = waveforms.shape
        if waveforms.dtype.kind != "f":
            waveforms = waveforms.astype(np.float64)

        # Get the norm (default is range bins)
        norm = n_range_bins if self.cfg.norm_is_range_bin else 1.0

        # Init output parameters
        pulse_peakiness = np.full(n_records, np.nan)

        # Compute peakiness for each waveform
        for i in np.arange(n_records):
            waveform = waveforms[i, self.cfg.skip_first_range_bins:]
            pulse_peakiness[i] = self._compute(waveform, norm)

        return pulse_peakiness

    def compute_for_waveform(self, waveform: npt.NDArray) -> float:
        """
        Compute pulse peakiness for a single waveform

        :param waveform:

        :return: pulse peakiness
        """

        # Get the waveform
        n_range_bins = waveform.shape
        if waveform.dtype.kind != "f":
            waveform = waveform.astype(np.float64)
        waveform = waveform[self.cfg.skip_first_range_bins:]

        # Get the norm (default is range bins)
        norm = n_range_bins if self.cfg.norm_is_range_bin else 1.0

        return self._compute(waveform, norm)

    @staticmethod
    def _compute(waveform: npt.NDArray, norm: Union[int, float]) -> float:
        """
        Compute pulse peakiness for a single waveform

        :param waveform: The waveform
        :param norm

        :return: pulse peakiness
        """
        try:
            pulse_peakiness = bn.nanmax(waveform) / (bn.nansum(waveform)) * norm
        except ZeroDivisionError:
            pulse_peakiness = np.nan
        return pulse_peakiness

    @property
    def required_options(self):
        return ["skip_first_range_bins", "norm_is_range_bin"]

