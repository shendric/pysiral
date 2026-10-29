# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"



from dataclasses import dataclass
from functools import partial
from pydantic import BaseModel
from typing import Tuple
from scipy.stats import linregress

import multiprocessing
import multiprocessing.pool

import bottleneck as bn
import numpy as np

from pysiral import PSRLCFG
from pysiral.l1 import L1Algorithm
from pysiral.l1.l1_data import Level1bData


@dataclass
class WFMTrailingEdgeSlopeData:
    waveform_normed: np.ndarray
    trailing_edge_start_idx: int
    trailing_edge_idx: np.ndarray
    noise_floor_power_threshold: float
    linregress_result: "LinregressResult"

    @property
    def range_gate_idx(self) -> np.ndarray:
        return np.arange(self.waveform_normed.size)

    @property
    def trailing_edge_slope(self) -> float:
        return self.linregress_result.slope if self.linregress_result is not None else np.nan

    @property
    def trailing_edge_slope_quality(self) -> float:
        return self.linregress_result.rvalue if self.linregress_result is not None else np.nan

    def debug_plot(self):
        x_te = self.range_gate_idx[self.trailing_edge_start_idx:]
        import matplotlib.pyplot as plt
        plt.figure(dpi=150)
        plt.plot(self.range_gate_idx, self.waveform_normed)
        plt.scatter(self.range_gate_idx[self.trailing_edge_idx], self.waveform_normed[self.trailing_edge_idx])
        plt.plot(x_te, self.linregress_result.intercept + self.trailing_edge_slope * x_te, color="red")
        plt.title(f"slope={self.trailing_edge_slope:.4f}, rvalue={self.trailing_edge_slope_quality:.3f}")
        plt.ylim(0, 1.1)
        plt.show()

class TrailingEdgeSlopeConfig(BaseModel):
    """
    Configuration class for the trailing edge slope algorithm.
    """
    noise_floor_power_threshold: float = 0.15


class L1PTrailingEdgeSlope(L1Algorithm):

    def __init__(self, **cfg) -> None:
        self.cfg = TrailingEdgeSlopeConfig(**cfg)

    def apply(self, l1: Level1bData):
        """

        :param l1:
        :return:
        """

        tes, tesq = self._get_trailing_edge_slope(
            l1.waveform.power,
            l1.classifier.first_maximum_index
        )
        l1.classifier.add(tes, "trailing_edge_slope")
        l1.classifier.add(tesq, "trailing_edge_slope_quality")

    def _get_trailing_edge_slope(
        self,
        waveforms: np.ndarray,
        first_maximum_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param waveforms:
        :param first_maximum_indices:
        :param use_multiprocessing:
        :return:
        """
        fit_inputs = zip(np.vsplit(waveforms, waveforms.shape[0]), first_maximum_indices)
        pool = multiprocessing.Pool(PSRLCFG.CPU_COUNT)
        results = pool.starmap(
            partial(
                trailing_edge_slope, noise_floor_power_threshold=self.cfg.noise_floor_power_threshold
            ),
            fit_inputs
        )
        pool.close()
        pool.join()
        output = [(r.trailing_edge_slope, r.trailing_edge_slope_quality) for r in results]
        return zip(*output)


def trailing_edge_slope(
        waveform: np.ndarray,
        trailing_edge_start_index: int,
        noise_floor_power_threshold: float = 0.15
) -> WFMTrailingEdgeSlopeData:
    """
    Compute trailing edge slope as linear fit to the trailing edge
    plus the quality (Pearson correlation factor) of the fit

    :param waveform: Waveform
    :param trailing_edge_start_index: Prior information of the
        start of the trailing edge
    :param noise_floor_power_threshold: Trailing edge power below this
        threshold will be ignored for the fit

    :return:
    """

    # Make slope parameter comparable by using normed waveform
    waveform_normed = np.ravel(waveform) / bn.nanmax(waveform)
    x = np.arange(waveform.size)

    trailing_edge_idx = np.logical_and(
        x >= trailing_edge_start_index,
        waveform_normed >= noise_floor_power_threshold
    )

    te_below_power_threshold = np.where(
        np.logical_and(
            x > trailing_edge_start_index,
            np.logical_not(trailing_edge_idx)
        )
    )[0]
    if te_below_power_threshold.any():
        trailing_edge_idx[te_below_power_threshold[0]:] = False

    # Fit may fail for various reasons
    try:
        result = linregress(x[trailing_edge_idx], waveform_normed[trailing_edge_idx])
    except ValueError:
        result = None

    return WFMTrailingEdgeSlopeData(
        waveform_normed,
        trailing_edge_start_index,
        trailing_edge_idx,
        noise_floor_power_threshold,
        result
    )

