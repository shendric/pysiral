# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


import multiprocessing
import multiprocessing.pool
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Tuple, Literal

import bottleneck as bn
import numpy as np
import numpy.typing as npt
from loguru import logger
from schema import And, Schema
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import argrelmin


from pysiral import PSRLCFG
from pysiral.core.clocks import StopWatch
from pysiral.core.functions import inverse_power
from pysiral.l1.l1_data import Level1bData
from pysiral.l1.alg import L1PProcItem


@dataclass
class WFMTrailingEdgeData:
    """ Data Class for L1PTrailingEdgeProperties """

    # Input parameters
    waveform_full: np.ndarray
    trailing_edge_start_idx: int
    waveform_trailing_edge_idx: np.ndarray
    trailing_edge_lower_envelope_idx: np.ndarray

    # Decay Parameters
    trailing_edge_fit: np.ndarray = field(default=None)
    trailing_edge_decay: float = field(default=np.nan)
    trailing_edge_decay_fit_quality: float = field(default=np.nan)
    trailing_edge_decay_mean_absolute_difference: float = field(default=np.nan)

    def decay_debug_plot(self, title="") -> None:

        import matplotlib.pyplot as plt
        x = np.arange(self.waveform_full.size)
        plt.figure(dpi=150)

        # Plot Waveform
        plt.plot(x, self.waveform_full, label="Waveform")

        # Plot Trailing Edge Lower Envelope
        idx_lower_envelope = self.trailing_edge_lower_envelope_idx + self.trailing_edge_start_idx
        plt.scatter(
            idx_lower_envelope,
            self.waveform_full[idx_lower_envelope],
            color="none", edgecolors="red", label="Trailing Edge Lower Envelope"
        )

        # Plot Fit
        if self.trailing_edge_fit is not None:
            plt.plot(
                self.waveform_trailing_edge_idx,
                self.trailing_edge_fit
            )
        plt.legend()
        plt.xlabel("Range Gate")
        plt.ylabel("Waveform Power Units")
        plt.title(title)
        plt.show()

    @property
    def trailing_edge_size(self) -> int:
        return self.waveform_trailing_edge_idx.size

    @property
    def waveform_trailing_edge_subset(self) -> np.ndarray:
        return self.waveform_full[self.waveform_trailing_edge_idx]

    @property
    def waveform_trailing_edge_subset_lower_envelope(self) -> np.ndarray:
        return self.waveform_trailing_edge_subset[self.trailing_edge_lower_envelope_idx]

    @property
    def first_maximum_power(self) -> float:
        return self.waveform_full[self.trailing_edge_start_idx]

    @property
    def decay_parameters(self) -> Tuple[float, float, float]:
        return (
            self.trailing_edge_decay,
            self.trailing_edge_decay_fit_quality,
            self.trailing_edge_decay_mean_absolute_difference
        )


class L1PTrailingEdgeProperties(L1PProcItem):

    def __init__(self, **cfg) -> None:
        super(L1PTrailingEdgeProperties, self).__init__(**cfg)
        # TODO: To be implemented as a Pydantic v2 data model
        self.config_validation_schema.validate(cfg)

    def apply(self, l1: Level1bData) -> None:
        """
        Compute late tail to peakiness and add to l1 data object

        :param l1: The Level-1 data container

        :return:
        """

        # Get configuration
        valid_first_maximum_index_range = self.cfg.get("valid_first_maximum_index_range")
        trailing_edge_width_kwargs = self.cfg.get("trailing_edge_width")

        timer = StopWatch().start()

        # Loop over all waveforms and compute parameters
        wfm = l1.waveform.power
        result = WaveFormTrailingEdgeParameter(
            wfm,
            l1.classifier.first_maximum_index,
            valid_first_maximum_index_range,
            trailing_edge_width_kwargs
        )

        timer.stop()
        logger.debug(f"- Trailing edge properties computed in {timer.get_seconds():.3f} seconds")
        logger.debug(f"- Trailing edge decay fit has failed for {result.params.decay_fit_has_failed.sum()} waveforms")

        l1.classifier.add(result.params.ted, "trailing_edge_decay")
        l1.classifier.add(result.params.tedfq, "trailing_edge_decay_fit_quality")
        l1.classifier.add(result.params.temad, "trailing_edge_mean_absolute_deviation")
        l1.classifier.add(result.params.tew, "trailing_edge_width")
        l1.classifier.add(result.params.teq, "trailing_edge_quality")
        l1.classifier.add(result.params.tepr, "trailing_edge_power_ratio")
        l1.classifier.add(result.params.te_exp_popt0, "trailing_edge_fit_popt0")
        l1.classifier.add(result.params.te_exp_popt1, "trailing_edge_fit_popt1")
        l1.classifier.add(result.params.te_exp_popt2, "trailing_edge_fit_popt2")

    @staticmethod
    def _compute_single_processing(*args) -> "WaveFormTrailingEdgeParameterData":
        """
        Compute waveform trailing edge parameter on a single thread

        :param args: Arguments to WaveFormTrailingEdgeParameter

        :return: Waveform trailing edge parameter object
        """
        return WaveFormTrailingEdgeParameter(*args).params

    # @staticmethod
    # def _compute_multi_processing(
    #         waveforms: npt.NDArray,
    #         valid_first_maximum_index_range: List,
    #         trailing_edge_width_kwargs: Dict
    # ) -> "WaveFormTrailingEdgeParameterData":
    #     """
    #     Compute waveform trailing edge parameter on a single thread
    #
    #     :param args: Arguments to WaveFormTrailingEdgeParameter
    #
    #     :return: Waveform trailing edge parameter object
    #     """
    #
    #     chunk_index_list, n_processes = get_1d_array_multiprocessing_chunks(waveforms.shape[0])
    #     logger.info(f"- Use multi-processing with {n_processes} workers")
    #     logger.disable("pysiral")
    #     with multiprocessing.pool.ThreadPool(n_processes) as process_pool:
    #         result = process_pool.map(
    #             WaveFormTrailingEdgeParameter,
    #             args=(
    #                 waveforms,
    #                 valid_first_maximum_index_range,
    #                 trailing_edge_width_kwargs
    #             )
    #         )
    #     # process_pool.close()
    #     # process_pool.join()
    #     # results = [job.get() for job in jobs]
    #     logger.enable("pysiral")
    #     #
    #     # params = results[0].params
    #     # for result in results[1:]:
    #     #     params.add(result.params)
    #     return params

    @property
    def config_validation_schema(self) -> Schema:

        return Schema(
            {
                "use_multiprocessing": bool,
                "valid_first_maximum_index_range": And(
                    list,
                    lambda x: all(item > 0 for item in x)
                ),
                "trailing_edge_width": {
                    "trailing_edge_end_power_treshold_normed": And(
                        float,
                        lambda x: 0.0 <= x <= 1.0,
                    ),
                    "oversample_factor": And(
                        int,
                        lambda x: x > 0
                    )
                },
            },
            error=f"Invalid config for {self.__class__.__name__}:\n "+"{}"
        )


class WaveFormTrailingEdgeParameterData(object):

    def __init__(self, n_records: int) -> None:
        """

        :param n_records:
        """
        # TODO: better use a pandas Dataframe etc.
        self.tew = np.full(n_records, np.nan)  # Trailing Edge Width
        self.teq = np.full(n_records, np.nan)  # Trailing Edge Quality
        self.ted = np.full(n_records, np.nan)  # Trailing Edge Decay (Müller et al. 2023)
        self.tedfq = np.full(n_records, np.nan)  # Trailing Edge Decay Fit Quality (Müller et al. 2023)
        self.tepr = np.full(n_records, np.nan)  # Trailing Edge Power Ratio
        self.temad = np.full(n_records, np.nan)
        self.decay_fit_has_failed = np.full(n_records, True)
        self.te_exp_popt0 = np.full(n_records, np.nan)
        self.te_exp_popt1 = np.full(n_records, np.nan)
        self.te_exp_popt2 = np.full(n_records, np.nan)

    def add(self, other_params: "WaveFormTrailingEdgeParameterData") -> None:
        self.tew = np.append(self.tew, other_params.tew)
        self.teq = np.append(self.teq, other_params.teq)
        self.ted = np.append(self.ted, other_params.ted)
        self.tedfq = np.append(self.tedfq, other_params.tedfq)
        self.tepr = np.append(self.tepr, other_params.tepr)
        self.temad = np.append(self.temad, other_params.temad)
        self.decay_fit_has_failed = np.append(self.decay_fit_has_failed, other_params.decay_fit_has_failed)
        self.te_exp_popt0 = np.append(self.tew, other_params.te_exp_popt0)
        self.te_exp_popt1 = np.append(self.tew, other_params.te_exp_popt1)
        self.te_exp_popt2 = np.append(self.tew, other_params.te_exp_popt2)


class WaveFormTrailingEdgeDecayFit(object):

    def __init__(self, fit_kwargs=None):
        self.default_fit_kwargs = {
            "method": "trf",
            "ftol": 1e-2,
            "gtol": 1e-2,
            "xtol": 1e-3,
            "loss": "linear"
        }
        self.fit_kwargs = self.default_fit_kwargs
        self.default_bounds = ([0.0, 0.0, -2000.], [1e6, 5.0, 1000.])
        self.p0 = None
        if isinstance(fit_kwargs, dict):
            self.fit_kwargs.update(fit_kwargs)

    def fit(self, data):
        p0 = [np.float64(data.first_maximum_power), np.float64(0.75), np.float64(0.0)]
        lb, ub = tuple(self.default_bounds)
        lb[0] = 0.9 * np.float64(data.first_maximum_power)
        ub[0] = 1.1 * np.float64(data.first_maximum_power)

        inverse_power_partial = partial(inverse_power, scale=np.float64(data.first_maximum_power))

        full_output = curve_fit(
            inverse_power_partial,
            data.trailing_edge_lower_envelope_idx.astype(np.float64) + 1.0,
            data.waveform_trailing_edge_subset_lower_envelope,
            p0=p0[1:],
            bounds=(lb[1:], ub[1:]),
            full_output=True,
            **self.fit_kwargs
        )

        popt = [*full_output[0], np.float64(data.first_maximum_power)]

        # import matplotlib.pyplot as plt
        # x = np.arange(max(data.trailing_edge_lower_envelope_idx)+1)
        # plt.figure()
        # plt.title(f"{popt=}")
        # x1 = data.waveform_trailing_edge_idx
        # plt.scatter(x1 - data.trailing_edge_start_idx + 1, data.waveform_full[x1],
        #             s=1, color="black")
        # plt.scatter(
        #     data.trailing_edge_lower_envelope_idx.astype(np.float64) + 1.0,
        #     data.waveform_trailing_edge_subset_lower_envelope,
        # )
        # plt.plot(x, inverse_power(x, *popt), label="fit")
        # plt.show()

        return popt


class WaveFormTrailingEdgeParameter(object):

    def __init__(self,
                 waveforms: npt.NDArray,
                 first_maximum_index: npt.NDArray,
                 valid_first_maximum_index_range: List,
                 trailing_edge_width_kwargs: Dict
                 ):
        """
        Algorithm class to compute trailing edge parameter

        :param waveforms: An array of waveforms.
        :param valid_first_maximum_index_range: F
        :param trailing_edge_width_kwargs:

        :raises None:
        """

        # Input data and config
        self.waveforms = waveforms
        self.first_maximum_index = first_maximum_index
        self.valid_first_maximum_index_range = valid_first_maximum_index_range
        self.trailing_edge_width_kwargs = trailing_edge_width_kwargs

        # Fit properties
        self.last_fit_popt = None

        # Result property
        self.params = WaveFormTrailingEdgeParameterData(waveforms.shape[0])

        # Compute the parameter
        self._compute()

    def _compute(self) -> None:
        """
        Loop over all waveforms and compute the parameters
        # TODO: Need documentation and potentially move to sub-modules
        """

        waveform_index_list = []
        waveform_data_stack = []
        num_waveforms = self.waveforms.shape[0]
        for i in np.arange(num_waveforms):

            # Collect data and sanity check
            trailing_edge_data = self.get_waveform_trailing_edge_data(
                self.waveforms[i, :],
                self.first_maximum_index[i]
            )
            if (
                self.valid_first_maximum_index_range[0] > trailing_edge_data.trailing_edge_start_idx or
                self.valid_first_maximum_index_range[1] < trailing_edge_data.trailing_edge_start_idx
            ):
                continue
            waveform_data_stack.append(trailing_edge_data)
            waveform_index_list.append(i)

        fit_class = WaveFormTrailingEdgeDecayFit()

        # single processing fit
        # results = []
        # for i in np.arange(len(waveform_data_stack)):
        #     popt = fit_class.fit(waveform_data_stack[i])
        #     results.append(popt)

        # multiprocessing fit
        pool = multiprocessing.Pool(PSRLCFG.CPU_COUNT)
        results = pool.imap(fit_class.fit, waveform_data_stack)
        pool.close()
        pool.join()
        results = list(results)

        # Note: Not all waveforms are fitted, e.g. if the trailing edge starts to
        #       late or too early. The list of results therefore might be shorter
        #       than the list of waveforms. `waveform_index_list` therefore maps
        #       the fit index to the original waveform index.
        for fit_index, i in enumerate(waveform_index_list):

            if results[fit_index] is None:
                continue
            popt = results[fit_index]
            self.params.te_exp_popt0[i] = popt[0]
            self.params.te_exp_popt1[i] = popt[1]
            self.params.te_exp_popt2[i] = popt[2]
            trailing_edge_data = self.get_trailing_edge_decay(waveform_data_stack[fit_index], popt)

            # In this case the fitting algorithm has failed
            # trailing_edge_data.decay_debug_plot("Trailing Edge fit")
            if trailing_edge_data.trailing_edge_fit is None:
                # trailing_edge_data.decay_debug_plot("Trailing Edge fit failed")
                continue
            self.params.ted[i], self.params.tedfq[i], self.params.temad[i] = trailing_edge_data.decay_parameters
            self.params.decay_fit_has_failed[i] = False

            # Compute other trailing edge properties
            self.params.tew[i] = self.get_trailing_edge_width(trailing_edge_data, **self.trailing_edge_width_kwargs)
            self.params.teq[i] = self.get_trailing_edge_quality(trailing_edge_data)
            self.params.tepr[i] = self.trailing_edge_power_ratio(trailing_edge_data)

    @staticmethod
    def get_waveform_trailing_edge_data(
            waveform_full: np.ndarray,
            first_maximum_index: int = None
    ) -> WFMTrailingEdgeData:
        """
        Aggregate all necesary trailing edge properties in a data class.

        :param waveform_full: The full waveform in original units
        :param first_maximum_index: Index of first maximum (definition of trailing edge start)

        :return: Data class
        """

        # NOTE: The index returned by `get_trailing_edge_lower_envelope_mask` describes
        #       the range gate index, but here the index relative to the first maximum is
        #       needed. Therefore, the index is shifted by the first maximum index.
        trailing_edge_lower_envelope_idx = get_trailing_edge_lower_envelope_mask(
            waveform_full.copy(),
            first_maximum_index,
            noise_level_normed=0.005,
            return_type="indices",
            mask_fill_value=False
        )
        trailing_edge_lower_envelope_idx -= first_maximum_index

        return WFMTrailingEdgeData(
            waveform_full,
            first_maximum_index,
            np.arange(first_maximum_index, waveform_full.size),
            trailing_edge_lower_envelope_idx,
        )

    @staticmethod
    def get_trailing_edge_decay(data: WFMTrailingEdgeData, popt: List) -> WFMTrailingEdgeData:
        """
        Fit inverse power law to the lower envelope of the trailing edge
        and compute decay factor, fit quality and residual to trailing
        edge power. Result will be added to data class.

        :param data: Pre-compiled trailing edge data
        :param popt: Fit coefficient (function inverse power low)

        :raises None:

        :return: Data class with updated results
        """

        # Compute fit
        x_values = np.arange(data.trailing_edge_size, dtype=np.float64) + 1
        data.trailing_edge_fit = inverse_power(x_values, *popt)

        # Compute parameters
        data.trailing_edge_decay = popt[1]  # second parameter in inverse power function
        data.trailing_edge_decay_mean_absolute_difference = bn.nanmedian(
            np.abs(data.trailing_edge_fit - data.waveform_trailing_edge_subset)
        )
        data.trailing_edge_decay_fit_quality = coeficient_of_determination(
            data.waveform_trailing_edge_subset,
            data.trailing_edge_fit
        )
        return data

    @staticmethod
    def trailing_edge_power_ratio(
            data: WFMTrailingEdgeData,
            noise_level_normed: float = 0.005
    ) -> float:
        """
        Compute the trailing edge power ratio defined as the fraction of the
        trailing edge power above a noise level of the exponentation fit.
        in relation to the integrated power of the full waveform.

        Valaues higher than 0.5 indicate that more than half of the
        waveform power is contained in the trailing edge above the noise level.

        :param data: Waveform Trailing Edge data object
        :param noise_level_normed: The noise level in relation to the first maximum power

        :return:
        """
        noise_level = noise_level_normed * data.first_maximum_power
        is_higher = data.waveform_trailing_edge_subset > (data.trailing_edge_fit + noise_level)
        try:
            return (
                np.nansum(data.waveform_trailing_edge_subset[is_higher] - data.trailing_edge_fit[is_higher]) /
                np.nansum(data.waveform_full)
            )
        except ZeroDivisionError:
            return np.nan

    @staticmethod
    def get_trailing_edge_width(
            data: WFMTrailingEdgeData,
            oversample_factor: int,
            trailing_edge_end_power_treshold_normed: float
    ) -> float:
        """
        Compute the width of the trailing edge based on the decay fit.

        :param data:
        :param oversample_factor:
        :param trailing_edge_end_power_treshold_normed:

        :raises None:

        :return: The trailing edge width in range gate units
        """

        # Get oversampled trailing edge fit
        range_gates_oversampled = np.linspace(
            0,
            data.trailing_edge_size - 1,
            data.trailing_edge_size * oversample_factor + 1,
        ) + data.waveform_trailing_edge_idx[0]
        range_gates_oversampled = np.around(range_gates_oversampled, 1)
        oversample_func = interp1d(data.waveform_trailing_edge_idx, data.trailing_edge_fit)
        trailing_edge_oversampled = oversample_func(range_gates_oversampled)

        # Get first (oversampled) range gate, where trailing edge is below
        # normed power threshold defining end of trailing edge
        trailing_edge_end_idx = np.where(
            trailing_edge_oversampled / data.first_maximum_power <= trailing_edge_end_power_treshold_normed
        )[0]

        if trailing_edge_end_idx.size == 0:
            # data.decay_debug_plot("Trailing Edge does not fall below minimum power threshold")
            return np.nan
        trailing_edge_end_range_gate = range_gates_oversampled[trailing_edge_end_idx][0]

        return trailing_edge_end_range_gate - data.trailing_edge_start_idx

    @staticmethod
    def get_trailing_edge_quality(data: WFMTrailingEdgeData) -> float:
        """

        :param data:
        :return:
        """
        power_diff = data.waveform_trailing_edge_subset[1:] - data.waveform_trailing_edge_subset[:-1]
        absolute_negative_power_diff = np.abs(power_diff[power_diff < 0.0])
        total_power_drop = bn.nansum(absolute_negative_power_diff)

        # Trailing edge quality indicator
        return total_power_drop / data.first_maximum_power

    @staticmethod
    def get_trailing_edge_lower_envelope(
            wfm_trailing_edge: npt.NDArray
    ) -> npt.NDArray:
        """

        :param wfm_trailing_edge:
        :return:
        """

        idx_le = argrelmin(wfm_trailing_edge)[0]
        idx_le = np.insert(idx_le, 0, 0)
        idx_le = np.insert(idx_le, len(idx_le), wfm_trailing_edge.size-1)

        wfm_change = [
            wfm_trailing_edge[idx_le[1:]] - wfm_trailing_edge[idx_le[:-1]]
        ][0]
        wfm_change = np.insert(wfm_change, 0, -1)
        return idx_le[wfm_change < 0.0]


def get_trailing_edge_lower_envelope_mask(
        waveform_power: np.ndarray,
        first_maximum_index: int,
        max_minimum_cleaning_passes: int = 5,
        noise_level_normed: float = 0.02,
        return_type: Literal["bool", "indices"] = "bool",
        mask_fill_value: bool = True
) -> np.ndarray:
    """
    Compute the indices of the waveforms trailing edge lower envelope.
    This method relies on the computation of relative minima via scipy.
    with added filtering.

    :param waveform_power: Waveform power values (any unit)
    :param first_maximum_index: Range gate index of the first maximum.
    :param max_minimum_cleaning_passes: Local minima on the trailing edge
        may not be in strictly decreasing in power depending on the
        off-nadir backscatter contanimation of the trailing edge.
        Local minima are iteratively removed that are not decreasing in
        power.
    :param noise_level_normed: Points within the noise level of the linear
        fit between successive local minima are added to the trailing edge
        mask. Noise level unit is the first maximum power.
    :param return_type: Options are a boolean array with the same dimensions
        as the waveform (default), or an index array.
    :param mask_fill_value: Default flag value before trailing edge (Default: True)

    :return:
    """

    # Perform operation on trailing edge only normalized by
    # first maximum power (trailing edge is defined as
    # everything after the first maximum).
    wfm_trailing_edge = np.array(waveform_power[first_maximum_index:])
    wfm_trailing_edge /= np.nanmax(wfm_trailing_edge)

    # Get local minima and add first maximum and last range gate
    # the list (important for fitting waveform power).
    idx_lmin = argrelmin(wfm_trailing_edge)[0]
    idx_lmin = np.insert(idx_lmin, 0, 0)
    idx_lmin = np.insert(idx_lmin, len(idx_lmin), wfm_trailing_edge.size - 1)

    # Throw out local minima that have greater power then the previous
    # one in several iterations
    for _ in np.arange(max_minimum_cleaning_passes):
        wfm_power_change = [
            wfm_trailing_edge[idx_lmin[1:]] - wfm_trailing_edge[idx_lmin[:-1]]
        ][0]
        wfm_change = np.insert(wfm_power_change, 0, -1)
        valid_local_minima = wfm_change < 0.0
        if valid_local_minima.all():
            break
        idx_lmin = idx_lmin[wfm_change < 0.0]

    # Create a leading edge mask. This is required for the
    # following operation
    mask_le = np.full(wfm_trailing_edge.size, mask_fill_value)
    mask_le[idx_lmin] = True

    # Take in again waveform points that are within the noise level
    # (ensures that are enough points for the fit, without deviating
    # from the lower envelope too much).
    # This is done by computing a linear fit between successive
    # local minima and checking the difference of actual power
    # values to the linear fit.
    # n_range_gates = len(waveform_power)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(wfm_trailing_edge)
    # plt.scatter(idx_lmin, wfm_trailing_edge[idx_lmin], color="red")
    # plt.show()

    is_higher = np.zeros(idx_lmin.size, dtype=bool)
    for i in np.arange(idx_lmin.size - 1):
        power_threshold = float(wfm_trailing_edge[idx_lmin[i]]) + noise_level_normed
        is_higher[i:] = np.logical_or(
            is_higher[i:],
            wfm_trailing_edge[idx_lmin[i:]] >= power_threshold
        )
        #
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.axhline(power_threshold, color="red")
        # plt.plot(wfm_trailing_edge, lw=0.4, color="black")
        # plt.scatter(idx_lmin[i], wfm_trailing_edge[idx_lmin[i]], color="blue", s=80)
        # plt.scatter(idx_lmin, wfm_trailing_edge[idx_lmin], c=is_higher, s=20)
        # plt.show()

    mask_le[idx_lmin[is_higher]] = False
    mask = np.full(waveform_power.size, False)
    mask[first_maximum_index:] = mask_le

    # import matplotlib.pyplot as plt
    # x = np.arange(waveform_power.size)
    # plt.figure()
    # plt.plot(waveform_power, lw=0.4, color="black")
    # plt.scatter(x[mask], waveform_power[mask], color="green", s=80)
    # plt.show()

    #
    #     # --- Method 1: Distance to POCA of linear fit ---
    #     idx_lmin_scaled = idx_lmin.astype(float) / n_range_gates
    #     k = wfm_trailing_edge[idx_lmin[i]]
    #     m = ((wfm_trailing_edge[idx_lmin[i+1]] - wfm_trailing_edge[idx_lmin[i]]) /
    #          (idx_lmin_scaled[i+1] - idx_lmin_scaled[i]))
    #
    #     for x, idx in enumerate(np.arange(idx_lmin[i]+1, idx_lmin[i+1])):
    #
    #         x0 = float(x + 1) / n_range_gates
    #         y0 = wfm_trailing_edge[idx]
    #
    #         x_poca = (x0 + m * y0 - m * k) / (m ** 2. + 1)
    #         y_poca = m * (x0 + m * y0 - m * k) / (m ** 2. + 1) + k
    #
    #         dist = np.sqrt((x_poca - x0)**2. + (y_poca - y0)**2)
    #         mask_le[idx] = dist <= noise_level_normed
    #
    # # Final step: Pad trailing edge mask to full waveform mask
    # mask = np.full(waveform_power.size, False)
    # mask[first_maximum_index:] = mask_le

    return mask if return_type == "bool" else np.where(mask)[0]


def coeficient_of_determination(y: np.ndarray, y_fit: np.ndarray) -> float:
    """
    Compute coeficient of determination

    :param y:
    :param y_fit:

    :return:
    """
    # residual sum of squares
    ss_res = bn.nansum((y - y_fit) ** 2)

    # total sum of squares
    ss_tot = bn.nansum((y - bn.nanmean(y)) ** 2)

    return 1 - (ss_res / ss_tot)
