# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = ["MarginalIceZoneFilterFlag", "SAMOSAMarginalIceZoneFilterFlag"]

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from loguru import logger

from pysiral.core.flags import ANDCondition
from pysiral.l2 import L2Algorithm


@dataclass
class MarginalIceZoneFilterData:
    """
    Data class for MarginalIceZoneFilterFlag
    NOTE: There will be several instances of this class per trajectory
          if the ice edge is crossed multiple times
    """

    # Filter configuration
    options: dict

    # The index of the ice edge
    ice_edge_idx: int

    # Flag indicating on which side of the ice edge index the
    # sea ice zone is located
    sea_ice_is_left: bool

    # Indices of the ocean waveforms for the computations
    # of the ice ocean leading edge widths at the ice edge
    ocean_idxs: np.ndarray

    # Indices of sea ice waveforms applicable for the
    # filter (important when there are several ice edges
    # in the trajectory)
    seaice_idxs: np.ndarray

    # Source leading edge
    leading_edge_width: np.ndarray

    # Source freeboard
    sea_ice_freeboard: np.ndarray

    # Source sea ice concentration
    sea_ice_concentration: np.ndarray

    # Source distance to ocean
    distance_to_ocean: np.ndarray

    # The average footprint spacing used to convert
    # physical filter sizes to array ranges
    footprint_spacing: float

    # Leading edge value of ocean waveforms at ice edge
    ocean_leading_edge_width_at_ice_edge: float = None

    # Filtered sea ice freeboard (for gradient computation)
    sea_ice_freeboard_filtered: np.ndarray = None

    # Sea ice freeboard gradient
    sea_ice_freeboard_gradient: np.ndarray = None

    # Filter flag value
    filter_flag: np.ndarray = None

    def debug_plot(self):
        """
        Create an overview plot of the filter result
        :return:
        """
        import matplotlib.pyplot as plt
        plt.figure(dpi=150)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        x = np.arange(self.sea_ice_freeboard_filtered.shape[0])

        ax1 = plt.subplot(2, 2, 1)

        filter_flag2_idx = np.where(self.filter_flag == 2)[0]
        if len(filter_flag2_idx) > 0:
            ax1.scatter(x[filter_flag2_idx],
                        self.sea_ice_freeboard_filtered[filter_flag2_idx],
                        s=24, linewidths=0, color="red")
        ax1.scatter(x[self.seaice_idxs],
                    self.sea_ice_freeboard_filtered[self.seaice_idxs],
                    s=8, linewidths=0, color="blue")
        ax1.scatter(x, self.sea_ice_freeboard, s=1, linewidths=0, color="black")

        ax1.plot(x, self.sea_ice_freeboard_filtered, lw=1, color="black", alpha=0.5)
        ax1.set_xlim(0, self.n_records)

        ax2 = plt.subplot(2, 2, 2)
        ax2.scatter(x, self.leading_edge_width, s=1, linewidths=0)
        ax2.scatter(x[self.ocean_idxs], self.leading_edge_width[self.ocean_idxs],
                    s=8, linewidths=0)
        ax2.axhline(self.ocean_leading_edge_width_at_ice_edge, lw=0.5, ls="--")
        ax2.set_xlim(0, self.n_records)

        ax3 = plt.subplot(2, 2, 3)
        ax3.scatter(x, self.sea_ice_freeboard_gradient, s=1, linewidths=0)
        ax3.axhline(0.0, ls="--", color="black", lw=0.5)
        ax3.set_xlim(0, self.n_records)

        ax4 = plt.subplot(2, 2, 4)
        ax4.scatter(x, self.filter_flag, s=1, linewidths=0)
        ax4.set_xlim(0, self.n_records)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.axvline(self.ice_edge_idx, ls=":", color="black", lw=0.5)

        plt.show()

    @property
    def n_records(self) -> int:
        return self.sea_ice_freeboard.shape[0]

    @property
    def has_flag(self) -> bool:
        return False if self.filter_flag is None else np.any(self.filter_flag >= 1)


class MarginalIceZoneFilterFlag(L2Algorithm):
    """
    Create a flag value that can be used to filter freeboard/thickness values
    that are affected by surface waves penetrating the marginal ice zone.
    The filter flag is not applied to any geophysical variables, but added
    to the l2 data container.

    The flag can take the following values:

        0: not in marginal ice zone
        1: in marginal ice zone: light to none wave influence detected
        2: in marginal ice zone: wave influence detected

    The flag values depend on:

        - leading edge with of ocean waveforms at the ice edge
        - sea ice freeboard gradient as a function of distance to the ice edge

    Thresholds to determine the flag values need to be specified in the
    options of the Level2 processor definition file:

    -   module: filter
        pyclass: MarginalIceZoneFilterFlag
        options:
            sea_ice_filter_size_m: <freeboard smoothing filter size>.
            ocean_filter_size_m: <ocean>.
            freeboard_smoother_filter_size_m: 150000.
            sea_ice_freeboard_miz_gradient: 0.002
            leading_edge_width_ocean_value: <threshold for ocean leading width>

    """

    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, l1: "Level1bData", l2: "Level2Data") -> np.ndarray:
        """
        API method for Level2ProcessorStep subclasses. Computes and add the filter flag to the l2 data object
        :param l1:
        :param l2:
        :return: Error flag array
        """

        # Get the default filter flag
        filter_flag_miz_error = self.get_clean_error_status(l2.n_records)

        # Get the output flag format
        # (Can be boolean (0, 1) or extented (-1, 0, 1, 2))
        boolean_flag_values = self.cfg.get("boolean_flag_values", False)

        # Only compute the filter flag, if all basic conditions are met
        filter_execute_conditions = [
            np.isfinite(l2.frb[:]).any(),       # There needs to be freeboard data
            np.isfinite(l2.footprint_spacing)   # There have been instances where footprint was NaN
        ]
        if not all(filter_execute_conditions):
            l2.set_auxiliary_parameter(
                "fmiz",
                "flag_miz",
                self.get_default_filter_flag(l2.n_records, boolean_flag_values),
                None)
            return filter_flag_miz_error

        # Compute the flag and store result in the L2 data container
        args = [l2.get_parameter_by_name("leading_edge_width"),
                l2.get_parameter_by_name("leading_edge_width_rolling_mean"),
                l2.get_parameter_by_name("pulse_peakiness_rolling_sdev"),
                l2.get_parameter_by_name("sea_ice_freeboard"),
                l2.get_parameter_by_name("sea_ice_concentration"),
                l2.get_parameter_by_name("distance_to_ocean"),
                l2.get_parameter_by_name("distance_to_low_ice_concentration"),
                l2.footprint_spacing]
        filter_flag, _ = self.get_miz_filter_flag(*args)

        # Simplify the filter to a true/false flag
        # The original flag and their modified values are:
        # -1: not in marginal ice zone
        # (-1 -> 0, 0 -> 0, 1 -> 0, 2 -> 1)
        if boolean_flag_values:
            filter_flag = np.where(filter_flag > 1, 1, 0).astype(np.byte)

        l2.set_auxiliary_parameter("fmiz", "flag_miz", filter_flag, None)
        return filter_flag_miz_error

    def get_miz_filter_flag(self,
                            leading_edge_width: np.ndarray,
                            leading_edge_width_rolling_mean: np.ndarray,
                            pulse_peakiness_rolling_sdev: np.ndarray,
                            sea_ice_freeboard: np.ndarray,
                            sea_ice_concentration: np.ndarray,
                            distance_to_ocean: np.ndarray,
                            distance_to_low_ice_concentration: np.ndarray,
                            footprint_spacing: float,
                            ) -> Tuple[np.ndarray, Union[None, List["MarginalIceZoneFilterData"]]]:
        """
        Compute the filter flag

        :param leading_edge_width:
        :param leading_edge_width_rolling_mean:
        :param pulse_peakiness_rolling_sdev:
        :param sea_ice_freeboard:
        :param sea_ice_concentration:
        :param distance_to_ocean:
        :param distance_to_low_ice_concentration:
        :param footprint_spacing:

        :return:
        """

        # The first filter step looks for ocean/ice transitions and
        # searches for freeboard anomalies
        filter_flag_01, miz_segments = self.get_miz_filter_flag_ice_edge_transition(
                leading_edge_width,
                sea_ice_concentration,
                sea_ice_freeboard,
                distance_to_ocean,
                footprint_spacing
        )

        # The second filter step checks if there are close passes
        # to the ice edge that are influenced by wave action
        # but never cross into open water. Such a pass will not
        # be detected by the first filter step
        filter_flag_02 = self.get_miz_filter_flag_proximity_based(
                distance_to_low_ice_concentration,
                sea_ice_concentration,
                leading_edge_width_rolling_mean,
                pulse_peakiness_rolling_sdev
        )

        # Merge the two filter_flags
        filter_flag = np.maximum(filter_flag_01, filter_flag_02)

        return filter_flag, miz_segments

    def get_miz_filter_flag_ice_edge_transition(
            self,
            leading_edge_width,
            sea_ice_concentration,
            sea_ice_freeboard,
            distance_to_ocean,
            footprint_spacing
    ) -> Tuple[np.ndarray, Union[None, List["MarginalIceZoneFilterData"]]]:
        """
        Get the filter flag for ice edge crossings

        :param leading_edge_width:
        :param sea_ice_concentration:
        :param sea_ice_freeboard:
        :param distance_to_ocean:
        :param footprint_spacing:

        :return:
        """

        # Get the initial filter_flag
        filter_flag = np.full(sea_ice_freeboard.shape[0], -1, dtype=int)

        # Step 1: Detect ice edge(s) in the trajectory data.
        ice_edge_segments = self.get_marginal_ice_zone_segments(
            sea_ice_concentration,
            distance_to_ocean,
            footprint_spacing
        )
        if ice_edge_segments is None:
            return filter_flag, ice_edge_segments

        # Step 2: Loop over the ice edge(s) and compute the filter flag
        miz_segments = []
        for ice_edge_segment in ice_edge_segments:

            ice_edge_idx, sea_ice_is_left, ocean_idxs, sea_ice_idxs = ice_edge_segment

            # Init the filter data class
            miz_segment = MarginalIceZoneFilterData(
                self.cfg.options,
                ice_edge_idx,
                sea_ice_is_left,
                ocean_idxs,
                sea_ice_idxs,
                leading_edge_width,
                sea_ice_freeboard,
                sea_ice_concentration,
                distance_to_ocean,
                footprint_spacing
            )

            self.compute_marginal_ice_zone_filter_flag(miz_segment)
            miz_segments.append(miz_segment)

            # Merge filter flag
            if not miz_segment.has_flag:
                continue

            for flag_value in [1, 2]:
                flag = ANDCondition()
                flag.add(miz_segment.filter_flag == flag_value)
                flag.add(filter_flag == -1)
                filter_flag[flag.indices] = flag_value

        return filter_flag, miz_segments

    def get_marginal_ice_zone_segments(self,
                                       sea_ice_concentration: np.ndarray,
                                       distance_to_ocean: np.ndarray,
                                       footprint_spacing: float
                                       ) -> Union[List, None]:
        """
        Find any transitions between sea ice / open water areas determined
        by sea ice concentration crossing the 15% threshold.
        :param sea_ice_concentration:
        :param distance_to_ocean:
        :param footprint_spacing:
        :return: List of indices marking the ice edge (e.g. first ice waveform).
        """

        # Get properties
        opt = self.cfg.options.get("ice_edge_transition", None)
        if opt is None:
            logger.error(f"{self.__class__.__name__}: Missing option category `ice_edge_transition`")
            return None

        sea_ice_filter_size_m = opt.get("sea_ice_filter_size_m", None)
        if sea_ice_filter_size_m is None:
            logger.error(f"{self.__class__.__name__}: Missing option `sea_ice_filter_size_m`")
            return None

        ocean_filter_size_m = opt.get("ocean_filter_size_m", None)
        if ocean_filter_size_m is None:
            logger.error(f"{self.__class__.__name__}: Missing option `ocean_leading_edge_window`")
            return None

        # Detect change ice edge by sea ice concentration jumps over the 15% threshold
        sic_flag = np.array(sea_ice_concentration > 15.0).astype(float)
        sic_flag_change = np.abs(np.ediff1d(sic_flag))
        ice_edge_candidate_idxs = np.where(sic_flag_change > 0.)[0]

        # Filter indices that are resulting of NaN values and return
        ice_edge_idxs = [
            idx
            for idx in ice_edge_candidate_idxs
            if np.all(np.isfinite([sea_ice_concentration[idx-1: idx+2]]))
        ]

        # Find out which side is which and get ocean/ice data range
        ocean_filter_size_float = ocean_filter_size_m / footprint_spacing
        ocean_filter_size = int(int(ocean_filter_size_float) // 2 * 2 + 1)

        sea_ice_filter_size_float = sea_ice_filter_size_m / footprint_spacing
        sea_ice_filter_size = int(int(sea_ice_filter_size_float) // 2 * 2 + 1)

        output = []
        for ice_edge_idx in ice_edge_idxs:

            sea_ice_is_left = sea_ice_concentration[ice_edge_idx+1] < 15.

            # sea ice indices are determined by the distance to the ice edge
            miz_sea_ice_segment = ANDCondition()
            miz_sea_ice_segment.add(distance_to_ocean <= sea_ice_filter_size_m)
            miz_sea_ice_segment.add(np.isfinite(sea_ice_concentration))
            idx_distance = np.abs(np.arange(sea_ice_concentration.shape[0]) - ice_edge_idx)
            miz_sea_ice_segment.add(idx_distance <= sea_ice_filter_size)
            sea_ice_idxs = miz_sea_ice_segment.indices

            # Ocean indices are determined by window next to the ice edge
            if sea_ice_is_left:
                ocean_idxs = np.arange(
                    ice_edge_idx+1,
                    min(ice_edge_idx + ocean_filter_size, sea_ice_concentration.shape[0])
                )
            else:
                ocean_idxs = np.arange(
                    max(ice_edge_idx - ocean_filter_size, 0),
                    ice_edge_idx+1
                )

            output.append((
                ice_edge_idx,
                sea_ice_is_left,
                ocean_idxs,
                sea_ice_idxs)
            )

        return output

    def compute_marginal_ice_zone_filter_flag(self, data: "MarginalIceZoneFilterData") -> None:
        """
        Compute the data properties in the marginal ice zone for the filter classification
        :param data:
        :return: None (miz prop is changed in-place)
        """

        # Get properties
        opt = self.cfg.options.get("ice_edge_transition", None)
        if opt is None:
            logger.error(f"{self.__class__.__name__}: Missing option category `ice_edge_transition")
            return None

        # First test: There must be a valid ocean leading edge width and freeboard estimate
        miz_frbs = data.sea_ice_freeboard[data.seaice_idxs]
        ocean_lews = data.leading_edge_width[data.ocean_idxs]
        if not miz_frbs.any() or not ocean_lews.any():
            return None

        # The leading edge value at the ocean side of the ice edge
        ocean_lew_at_ice_edge = np.nanmean(ocean_lews)
        data.ocean_leading_edge_width_at_ice_edge = ocean_lew_at_ice_edge

        # Compute the gradient of filtered sea ice freeboard as change / km
        # which is expected to be strong in the presence of wave influence
        x_all = np.arange(0, data.n_records)
        freeboard_smoother_filter_size_m = opt.get("freeboard_smoother_filter_size_m", None)
        if freeboard_smoother_filter_size_m is None:
            logger.error(f"{self.__class__.__name__}: Missing option `freeboard_smoother_filter_size_m")
            return None

        # Compute a smoothed & gap-less freeboard time series and its gradient
        freeboard_smoother_filter_size_float = freeboard_smoother_filter_size_m / data.footprint_spacing
        freeboard_smoother_filter_size = int(int(freeboard_smoother_filter_size_float) // 2 * 2 + 1)
        frb_flt, frb_flt_gradient = self.get_filtered_value_and_gradient(
            x_all,
            data.sea_ice_freeboard,
            freeboard_smoother_filter_size,
            gradient_scale_factor=1000./data.footprint_spacing
        )
        data.sea_ice_freeboard_filtered = frb_flt
        data.sea_ice_freeboard_gradient = frb_flt_gradient

        # Find the gradient value next to the ice edge
        valid_frb_idx = np.where(np.isfinite(frb_flt_gradient))[0]
        if len(valid_frb_idx) < 2:
            return

        next_to_ice_edge_idx = valid_frb_idx[-1] if data.sea_ice_is_left else valid_frb_idx[1]
        frb_gradient_at_ice_edge = frb_flt_gradient[next_to_ice_edge_idx]

        # Compute the filter flag
        filter_flag = np.full(data.n_records, 0).astype(int)

        # Filter flag 1: Is in marginal ice zone
        # All waveforms on the sea ice side within a specified distance to the ocean
        # NOTE: From SIC based distance_to_ocean, not the along-track distance
        filter_flag[data.seaice_idxs] = 1

        frb_gradient_threshold = opt.get("sea_ice_freeboard_miz_gradient")
        lew_ocean_threshold = opt.get("leading_edge_width_ocean_value")

        # Find first zero passing of freeboard gradient next to ice edge
        condition = (
                np.abs(frb_gradient_at_ice_edge) >= frb_gradient_threshold
                or
                ocean_lew_at_ice_edge >= lew_ocean_threshold
        )

        if condition:
            filter_flag_idx = self.get_impacted_range(frb_flt,
                                                      frb_flt_gradient,
                                                      data.ice_edge_idx,
                                                      data.sea_ice_is_left)
            if filter_flag_idx.shape[0] > 0:
                filter_flag[filter_flag_idx] = 2
        data.filter_flag = filter_flag

    @staticmethod
    def get_filtered_value_and_gradient(x: np.ndarray,
                                        y: np.ndarray,
                                        filter_size: int,
                                        gradient_scale_factor: float = 1.0,
                                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a filtered version and its gradient of a noisy and gappy variable
        (In this case likely freeboard)
        :param x:
        :param y:
        :param filter_size:
        :param gradient_scale_factor:
        :return:
        """

        # Step 1: Use a lowess filter to create
        y_filtered = lowess_smooth(x, y, filter_size)

        # Step 2: Fill NaN Gaps in gradient with linear interpolation
        y_filtered_gapless = interp1d_gap_filling(y_filtered)

        # Step 3: Remove minor wiggles which may happen in the lowess filtering
        #         Also ensure that the edges of the valid data are not flattened
        post_filter_size = 11
        y_filtered2 = astropy_smooth(y_filtered_gapless, post_filter_size, boundary="fill", fill_value=np.nan)
        valid_idx = np.where(np.isfinite(y_filtered2))[0]
        i0, i1 = valid_idx[0], valid_idx[-1]
        y_filtered2[i0:i0+post_filter_size] = y_filtered_gapless[i0:i0+post_filter_size]
        y_filtered2[i1-post_filter_size:i1+1] = y_filtered_gapless[i1-post_filter_size:i1+1]

        # Use scaling for gradient (meant to convert to m/m; m/km, etc).
        y_gradient = np.ediff1d(y_filtered2) * gradient_scale_factor
        y_gradient = np.insert(y_gradient, 0, np.nan)

        return y_filtered2, y_gradient

    @staticmethod
    def get_impacted_range(sea_ice_freeboard_filtered: np.ndarray,
                           sea_ice_freeboard_gradient: np.ndarray,
                           ice_edge_idx: int,
                           sea_ice_is_left: bool) -> np.ndarray:
        # sourcery skip: inline-immediately-returned-variable
        """
        Compute the impacted area which is defined as the range between the nearest point
        to the ice edge to the point where the sea ice freeboard gradient show the first
        zero crossing (-> reversal of freeboard trend). This method should only be called,
        when it is quite certain that the freeboard is impacted by wave.

        :param sea_ice_freeboard_filtered:
        :param sea_ice_freeboard_gradient:
        :param ice_edge_idx:
        :param sea_ice_is_left:
        :return: Indices Array (potentially empty)
        """

        # Compute zero crossings (and discard the last incorrect one)
        # NOTE: without the `np.isfinite` check there will be a lot false zero crossings.
        zero_crossing_idx = np.where(
            np.logical_and(
                np.diff(np.sign(sea_ice_freeboard_gradient)).astype(bool),
                np.isfinite(sea_ice_freeboard_gradient[:-1]),
            )
        )[0]
        if zero_crossing_idx.size <= 1:
            return np.array([])

        # Remove the last values (artefact of zero crossing estimation)
        zero_crossing_idx = zero_crossing_idx[:-1]

        # Try to estimate if the zero crossing of the freeboard gradient
        # is just "intermediary wiggle". The conditions for this is that
        # the magnitude of freeboard is generally still lower for the
        # next local minimum
        if sea_ice_is_left:
            max_impacted_range_idx = zero_crossing_idx[-1]
            candidate_idxs = np.arange(zero_crossing_idx.size-2, 0, -2)
        else:
            max_impacted_range_idx = zero_crossing_idx[0]
            candidate_idxs = np.arange(2, zero_crossing_idx.size, 2)
        for candidate_idx in candidate_idxs:
            new_candidate_idx = zero_crossing_idx[candidate_idx]
            if sea_ice_freeboard_filtered[max_impacted_range_idx] > sea_ice_freeboard_filtered[new_candidate_idx]:
                max_impacted_range_idx = new_candidate_idx
            else:
                break

        # Return a list of indices
        return (
            np.arange(max_impacted_range_idx, ice_edge_idx+1)
            if sea_ice_is_left else
            np.arange(ice_edge_idx, max_impacted_range_idx)
        )

    def get_miz_filter_flag_proximity_based(
            self,
            distance_to_low_ice_concentration: np.ndarray,
            sea_ice_concentration: np.ndarray,
            leading_edge_width_rolling_mean: np.ndarray,
            pulse_peakiness_rolling_sdev: np.ndarray
    ) -> Union[np.ndarray, None]:
        """
        Compute the MIZ Flag based on a set of thresholds only. Conditions

        :param distance_to_low_ice_concentration:
        :param sea_ice_concentration:
        :param leading_edge_width_rolling_mean:
        :param pulse_peakiness_rolling_sdev:
        :return:
        """

        filter_flag = np.full(sea_ice_concentration.shape[0], -1, dtype=int)

        # Get properties
        opt = self.cfg.options.get("ice_edge_proximity", None)
        if opt is None:
            logger.error(f"{self.__class__.__name__}: Missing option category `ice_edge_proximity")
            return None

        close_proximity_idx = np.where(
            np.logical_and(
                distance_to_low_ice_concentration <= opt["low_ice_concentration_distance_max"],
                np.isfinite(sea_ice_concentration)
            )
        )
        filter_flag[close_proximity_idx] = 1

        miz_threshold_filter = ANDCondition()
        miz_threshold_filter.add(filter_flag >= 1)
        miz_threshold_filter.add(sea_ice_concentration <= opt["sea_ice_concentration_max"])
        miz_threshold_filter.add(leading_edge_width_rolling_mean >= opt["leading_edge_width_rolling_mean_min"])
        miz_threshold_filter.add(pulse_peakiness_rolling_sdev <= opt["pulse_peakiness_rolling_sdev_max"])

        filter_flag[miz_threshold_filter.indices] = 2
        return filter_flag

    @staticmethod
    def get_default_filter_flag(n_records: int, boolean_flag_values: bool) -> np.ndarray:
        fill_value = 0 if boolean_flag_values else -1
        return np.full(n_records, fill_value).astype(np.byte)

    @property
    def l2_dependencies(self):
        return ["surface_type",
                "leading_edge_width",
                "sea_ice_freeboard",
                "distance_to_ocean"]

    @property
    def l2_output_vars(self):
        return ["flag_miz"]

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["filter"]


class SAMOSAMarginalIceZoneFilterFlag(L2Algorithm):
    """
    Create a flag value that can be used to filter freeboard/thickness values
    that are affected by surface waves penetrating the marginal ice zone based
    on output from the SAMOSA+ algorithm.

    The flag can take the following values:

        0: not in marginal ice zone
        1: in marginal ice zone: light to none wave influence detected
        2: in marginal ice zone: wave influence detected

    The flag values depend on:

        - leading edge with of ocean waveforms at the ice edge
        - sea ice freeboard gradient as a function of distance to the ice edge

    Thresholds to determine the flag values need to be specified in the
    options of the Level2 processor definition file:

    -   module: filter
        pyclass: SAMOSAMarginalIceZoneFilterFlag
        options:
            max_ocean_proximity: <maximum distance to ocean in meter>
            min_significant_waveheight: <minimum significant wave height in meter>

    """

    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, l1: "Level1bData", l2: "Level2Data") -> np.ndarray:
        """
        API method for Level2ProcessorStep subclasses. Computes and add the filter flag to the l2 data object

        :param l1:
        :param l2:

        :return: Error flag array
        """

        # Get the default filter flag
        filter_flag_miz_error = self.get_clean_error_status(l2.n_records)

        # Get input variables
        miz_filter = np.zeros((l2.n_records,)).astype(np.byte)
        try:
            ocean_proximity = l2.get_parameter_by_name("distance_to_ocean")
            significant_waveheight = l2.get_parameter_by_name("samosa_swh")
            sea_ice_concentration = l2.get_parameter_by_name("sea_ice_concentration")
        except KeyError:
            logger.warning(f"{self.__class__.__name__}: Missing input variables (did SAMOSA+ run?)")
            l2.set_auxiliary_parameter("fmiz", "flag_miz", miz_filter)
            return np.logical_not(filter_flag_miz_error)

        # Compute filter flag value
        conditions = (
            ocean_proximity <= self.cfg.options.max_ocean_proximity,
            significant_waveheight >= self.cfg.options.min_significant_waveheight,
            sea_ice_concentration >= 15.0
        )
        miz_filter_flag = np.logical_and.reduce(conditions)
        miz_filter[miz_filter_flag] = 1

        # Add parameter to the L2 data container (no uncertainties)
        l2.set_auxiliary_parameter("fmiz", "flag_miz", miz_filter)
        return filter_flag_miz_error

    @property
    def l2_dependencies(self):
        return [
            "surface_type",
            "samosa_swh",
            "samosa_leading_edge_error",
            "distance_to_ocean",
            "sea_ice_concentration"
        ]

    @property
    def l2_output_vars(self):
        return ["flag_miz"]

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["filter"]
