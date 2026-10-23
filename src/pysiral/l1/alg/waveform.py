# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 13:07:10 2016

@author: shendric
"""

import multiprocessing
import multiprocessing.pool
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Literal, Tuple, Union

import bottleneck as bn
import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel
from schema import And, Schema
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import argrelmin
from scipy.stats import linregress


from pysiral import PSRLCFG
from pysiral.core.clocks import StopWatch
from pysiral.core.flags import RadarModes
from pysiral.core.functions import inverse_power
from pysiral.l1 import L1Algorithm
from pysiral.l1.l1_data import Level1bData
from pysiral.l1.alg import L1PProcItem
from pysiral.retracker.tfmra import cTFMRA










class L1PLateTail2PeakPower(L1PProcItem):

    def __init__(self, **cfg) -> None:
        super(L1PLateTail2PeakPower, self).__init__(**cfg)

    def apply(self, l1: Level1bData) -> None:
        """
        Compute late tail to peakiness and add to l1 data object
        :param l1:

        :return:
        """

        # Init the classifier
        ltpp = np.full(l1.info.n_records, np.nan)

        # The power threshold depends on the radar mode
        late_tail_window_idx_dict = self.cfg.get("late_tail_window_idx", None)
        if late_tail_window_idx_dict is None:
            logger.error("settings `late_tail_window_idx` is not defined")
            return

        # Loop over all waveforms and compute ltpp
        wfm = l1.waveform.power
        radar_mode = l1.waveform.radar_mode
        invalid_radar_mode_encountered = 0

        for i in np.arange(wfm.shape[0]):
            late_tail_window_idx = late_tail_window_idx_dict.get(
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
        wfm_max_power_index = bn.nanargmax(wfm)
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




class L1PLeadingEdgePeakiness(L1PProcItem):

    def __init__(self, **cfg):
        super(L1PLeadingEdgePeakiness, self).__init__(**cfg)
        for option_name in self.required_options:
            if option_name not in self.cfg.keys():
                logger.error(f"Missing option: {option_name} -> Leading Edge Quality will not be computed")

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

        # Get the window for the pulse peakiness computation
        window_size = self.cfg.get("window_size", None)
        if window_size is None:
            logger.error("Option `window size` not available -> skipping leading edge peakiness")
            l1.classifier.add(lep, "leading_edge_peakiness")
            return

        # Loop over all waveforms
        for i in np.arange(wfm.shape[0]):
            if fmi[i] < 0:
                continue
            lep[i] = self.leading_edge_peakiness(wfm[i, :], fmi[i], window_size)

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

    @property
    def required_options(self):
        return ["window_size"]






class CS2PulsePeakiness(object):
    """
    Calculates Pulse Peakiness (full, left & right) for CryoSat-2 waveform
    counts
    XXX: This is a 1 to 1 legacy implementation of the IDL CS2AWI method,
         consistent method of L1bData or L2Data is required
    """

    def __init__(self, wfm_counts, pad=2):
        shape = np.shape(wfm_counts)
        self._n = shape[0]
        self._n_range_bins = shape[1]
        self._pad = pad
        self._peakiness = np.full(self._n, np.nan).astype(np.float32)
        self._peakiness_r = np.full(self._n, np.nan).astype(np.float32)
        self._peakiness_l = np.full(self._n, np.nan).astype(np.float32)
        self._noise_floor = np.full(self._n, np.nan).astype(np.float32)
        # self.peakiness_no_noise_removal = np.full(self._n, np.nan).astype(np.float32)
        self._peakiness_normed = np.full(self._n, np.nan).astype(np.float32)
        self._calc_parameters(wfm_counts)

    def _calc_parameters(self, wfm_counts):
        for i in np.arange(self._n):
            try:
                y = wfm_counts[i, :].flatten().astype(np.float32)
                self._noise_floor[i] = bn.nanmean(y[:11])
                # y_no_noise_removal = y.copy()
                y -= self._noise_floor[i]  # Remove Noise
                y[np.where(y < 0.0)[0]] = 0.0  # Set negative counts to zero
                yp = np.nanmax(y)  # Waveform peak value
                ypi = np.nanargmax(y)  # Waveform peak index
                if 3 * self._pad < ypi < self._n_range_bins - 4 * self._pad:
                    self._peakiness_l[i] = yp / bn.nanmean(y[ypi - 3 * self._pad:ypi - 1 * self._pad + 1]) * 3.0
                    self._peakiness_r[i] = yp / bn.nanmean(y[ypi + 1 * self._pad:ypi + 3 * self._pad + 1]) * 3.0
                    self._peakiness_normed[i] = yp / y.sum()
                    self._peakiness[i] = self.peakiness_normed[i] * self._n_range_bins

                    # self.peakiness_no_noise_removal[i] = np.nanmax(y_no_noise_removal) / y_no_noise_removal.sum() * self._n_range_bins
            except ValueError:
                self._peakiness_l[i] = np.nan
                self._peakiness_r[i] = np.nan
                self._peakiness[i] = np.nan

    @property
    def peakiness(self):
        return self._peakiness

    @property
    def peakiness_normed(self):
        return self._peakiness_normed

    @property
    def noise_floor(self):
        return self._noise_floor

    @property
    def peakiness_r(self):
        return self._peakiness_r

    @property
    def peakiness_l(self):
        return self._peakiness_l


# Late tail to peak power (LTPP) ratio added for fmi needs
class S3LTPP(object):
    """
    Calculates Late-Tail-to-Peak-Power ratio.
    source: Rinne 2016
    """

    def __init__(self, wfm_counts, pad=1):
        # Warning: if 0padding is introduced in S3 L1 processing baseline, pad must be set to 2
        shape = np.shape(wfm_counts)
        self._n = shape[0]
        self._n_range_bins = shape[1]
        self._pad = pad
        dtype = np.float32
        self._ltpp = np.ndarray(shape=[self._n], dtype=dtype) * np.nan
        self._calc_parameters(wfm_counts)

    def _calc_parameters(self, wfm_counts):
        # loop over the waveforms
        for i in np.arange(self._n):
            try:
                y = wfm_counts[i, :].flatten().astype(np.float32)
                y -= bn.nanmean(y[:11])  # Remove Noise
                y[np.where(y < 0.0)[0]] = 0.0  # Set negative counts to zero
                yp = np.nanmax(y)  # Waveform peak value

                if np.isnan(yp):  # if the current wf is nan
                    # no ltpp can be computed
                    self._ltpp[i] = np.nan
                else:
                    ypi = np.nanargmax(y)  # Waveform peak index

                    # gates to compute the late tail:
                    # [ypi+50:ypi+70] if 0padding=2, [ypi+25:ypi+35] if 0padding=1
                    gate_start = ypi + self._pad * 25
                    gate_stop = ypi + self._pad * 35 + 1

                    if gate_start > self._n_range_bins or gate_stop > self._n_range_bins:
                        # not enough gates to compute the LTPP
                        self._ltpp[i] = np.nan
                    else:
                        self._ltpp[i] = np.mean(y[gate_start:gate_stop]) / yp

            except ValueError:
                self._ltpp[i] = np.nan

    @property
    def ltpp(self):
        return self._ltpp


class CS2LTPP(object):
    """
    Calculates Late-Tail-to-Peak-Power ratio.
    """

    def __init__(self, wfm_counts, pad=2):
        shape = np.shape(wfm_counts)
        self._n = shape[0]
        self._n_range_bins = shape[1]
        self._pad = pad
        dtype = np.float32
        self._ltpp = np.ndarray(shape=[self._n], dtype=dtype) * np.nan
        self._calc_parameters(wfm_counts)

    def _calc_parameters(self, wfm_counts):
        # loop over the waveforms
        for i in np.arange(self._n):
            y = wfm_counts[i, :].flatten().astype(np.float32)
            y -= bn.nanmean(y[:11])  # Remove Noise
            y[np.where(y < 0.0)[0]] = 0.0  # Set negative counts to zero
            yp = np.nanmax(y)  # Waveform peak value
            ypi = bn.nanargmax(y)  # Waveform peak index

            # AMANDINE: implementation for wf of 256 bins (128 zero-padded)?
            onediv = float(1) / float(41)

            # AMANDINE: here i seems to be understood as gate index, but it is wf index!?
            if i == 256:
                break
            if [i > (ypi + 100)] and [i < (ypi + 140)]:  # AMANDINE: syntax to be checked
                try:
                    self._ltpp[i] = (onediv * float(y[i])) / float(yp)  # AMANDINE: where is the sum in this formula?
                except ZeroDivisionError:
                    self._ltpp[i] = np.nan

    @property
    def ltpp(self):
        return self._ltpp


class EnvisatWaveformParameter(object):
    """
    Currently only computes pulse peakiness for Envisat waveforms
    from SICCI processor.

    Parameter for Envisat from SICCI Processor
        skip = 5
        bins_after_nominal_tracking_bin = 83
    """

    def __init__(self, wfm, skip=5, bins_after_nominal_tracking_bin=83):
        self.t_n = bins_after_nominal_tracking_bin
        self.skip = skip
        self._n = wfm.shape[0]
        self._n_range_bins = wfm.shape[1]
        self._init_parameter()
        self._calc_parameter(wfm)

    def _init_parameter(self):
        self.peakiness_old = np.ndarray(shape=self._n, dtype=np.float32)
        self.peakiness = np.ndarray(shape=self._n, dtype=np.float32) * np.nan

    def _calc_parameter(self, wfm):

        for i in np.arange(self._n):
            # Discard first bins, they are FFT artefacts anyway
            wave = wfm[i, self.skip:]

            # old peakiness
            try:
                pp = 0.0 + self.t_n * float(max(wave)) / float(sum(wave))
            except ZeroDivisionError:
                pp = np.nan
            self.peakiness_old[i] = pp

            # new peakiness
            try:
                self.peakiness[i] = float(max(wave)) / float(sum(wave)) * self._n_range_bins
            except ZeroDivisionError:
                self.peakiness[i] = np.nan




