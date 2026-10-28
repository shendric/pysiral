# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:30:53 2016

@author: Stefan Hendricks
"""

from dataclasses import dataclass
from typing import List, Tuple, Union

import bottleneck as bn
import numpy as np
import statsmodels.api as sm
from astropy.convolution import Box1DKernel, convolve
from loguru import logger
from scipy.interpolate import interp1d

from pysiral.core.flags import FlagContainer, ORCondition
from pysiral.l2 import L2Algorithm


class L2ParameterValidRange(L2Algorithm):
    """
    Filters L2 variables that are outside a valid range.

    Usage in Level-2 processor definition files:

    -   module: filter
        pyclass: L2ParameterValidRange
        options:
            source_variable: <source_variable>
            target_variables: [<target_variables>]
            valid_minimum_point_value: <min_val>
            valid_maximum_point_value: <max_val>

    will lead to that all target_variables will be set to np.nan if either

        source_variable < min_val  of source_variable > max_val

    """

    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, l1, l2):
        """
        Mandatory method for Level2ProcessorStep will change l2 data object in place
        :param l1:
        :param l2:
        :return:
        """

        # Step 1: Get the filter flag
        parameter = getattr(l2, self.cfg.options.source_variable)
        invalid = ORCondition()
        invalid.add(parameter < self.cfg.options.valid_minimum_point_value)
        invalid.add(parameter > self.cfg.options.valid_maximum_point_value)
        filter_flag = FlagContainer(invalid.flag)

        # Check if any flag was raised
        if filter_flag.num == 0:
            return filter_flag.flag

        # Modify target variables
        for parameter_id in self.cfg.options.target_variables:
            var = getattr(l2, parameter_id)
            var.set_nan_indices(filter_flag.flag)
            setattr(l2, parameter_id, var)

        # The filter flag can be used as error_status
        return filter_flag.flag

    @property
    def l2_dependencies(self):
        input_vars = [self.cfg.options.source_variable]
        input_vars.extend(self.cfg.options.target_variables)
        return input_vars

    @property
    def l2_output_vars(self):
        return []

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["filter"]


class RemoveNonOceanData(L2Algorithm):
    """
    Ensure that data over non-ocean surfaces is set to NaN.

    Usage in Level-2 processor definition files:

    -   module: filter
        pyclass: RemoveNonOceanData
        options:
            target_variables: [<target_variables>]

    will lead to that all finite values of the target variables
    are set to NaN if the surface type is either land (6) or land ice (7).
    Any occurrence are logged.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, l1b: "Level1bData", l2: "Level2Data") -> np.ndarray:

        # Get the error flag
        error_status = self.get_clean_error_status(l2.n_records)

        # Check if the surface type indicates either land (7) or land ice (6).
        is_non_ocean = np.isin(l2.surface_type.flag, [6, 7])

        # Modify target variables
        parameter_names = self.cfg.options.get("target_variables", [])
        for parameter_name in parameter_names:

            var = l2.get_parameter_by_name(parameter_name)
            if var is None:
                msg = f"Variable {parameter_name} not found in Level-2 data object."
                self.error.add_error("filter-non-ocean-data", msg)
                error_status[:] = True
                continue

            filter_idxs = np.logical_and(np.isfinite(var[:]), is_non_ocean)
            if (num_land_values := np.sum(filter_idxs)) == 0:
                continue

            logger.info(f"- Remove non-ocean data from {parameter_name} ({num_land_values} records)")
            var.set_nan_indices(filter_idxs)
            setattr(l2, parameter_name, var)

        return error_status

    @property
    def l2_dependencies(self):
        return ["surface_type"]

    @property
    def l2_output_vars(self):
        return self.cfg.options.get("target_variables", [])

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["filter"]


class ParameterSmoother(L2Algorithm):
    """
    Creates a filtered/smoothed copy of a given parameter.

    Usage in Level-2 processor definition files:

    -   module: filter
        pyclass: ParameterSmoother
        options:
            source_variable: <source_variable>
            target_variable_name: <output variable auxiliary data name>
            target_variable_id: <output variable auxiliary data id>
            smoothing_method: <the method for the smoothing: box_filter|gaussian_process>
            smoother_args: dict
            retain_nan_mask: bool

    will register a new auxiliary variable output_variable_name|output_variable_id to the
    Level-2 data object
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, l1b, l2):
        """
        Compute the smoother
        :param l1b:
        :param l2:
        :return:
        """

        # Get the error flag
        error_status = self.get_clean_error_status(l2.n_records)

        # Get the output name and id of the resulting auxiliary data variable
        auxid = self.cfg.options.target_variable_id
        auxname = self.cfg.options.target_variable_name

        # Get a dictionary of the filter function
        filter_func = {
            "box_filter": self.box_filter_smoother,
            "lowess": self.lowess_smoother
        }

        # check if requested smoothing method is implemented
        if self.cfg.options.smoothing_method not in filter_func.keys():
            error_status[:] = True
            msg = f"Not-Implemented: Smoothing method: {self.cfg.options.smoothing_method}"
            self.error.add_error("filter-not-implemented", msg)
            l2.register_auxvar(auxid, auxname, np.full(error_status.shape, np.nan), None)
            return error_status

        # Get the source parameter
        var = getattr(l2, self.cfg.options.source_variable)
        var_filtered, _ = filter_func[self.cfg.options.smoothing_method](var[:], **self.cfg.options.smoother_args)

        # [Optional] Remove interpolated values from waveforms that are classified as land (flag value 6) or
        # land ice (flag value 7). This part of the algorithm is optional and must be explicitly activated
        # by setting `ocean_domain_only: True` in the config file to not break older configurations.
        ocean_domain_only = self.cfg.options.get("ocean_domain_only", False)
        if ocean_domain_only:
            is_non_ocean = np.isin(l2.surface_type.flag, [6, 7])
            logger.info(f"- Filter apply ocean-domain-only filter ({np.where(is_non_ocean)[0].size} records)")
            var_filtered[is_non_ocean] = np.nan

        # Add the result as auxiliary data set without an uncertainty
        l2.set_auxiliary_parameter(auxid, auxname, var_filtered)
        return error_status

    @staticmethod
    def box_filter_smoother(x, window=5):
        return astropy_smooth(np.array(x), window)

    @staticmethod
    def lowess_smoother(y, **smoother_args):
        lowess = sm.nonparametric.lowess
        x = np.arange(y.shape[0])

        # Compute the data fraction to use at each y-value
        data_fraction = min(smoother_args.get("filter_size_n_points", 1501.) / float(y.shape[0]), 1.)

        # Compute the lowess filter with potential keywords from the config file
        filter_props = smoother_args.get("filter_props", {})
        z = lowess(y, x, frac=data_fraction, return_sorted=False, **filter_props)

        return z, None

    @property
    def l2_dependencies(self):
        return [self.cfg.options.source_variable]

    @property
    def l2_output_vars(self):
        return [self.cfg.options.target_variable_id]

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["filter"]



def astropy_smooth(x, window, **kwargs):
    kernel = Box1DKernel(window)
    return convolve(x, kernel, **kwargs)


def interp1d_gap_filling(y: np.ndarray, **interp_kwargs) -> np.ndarray:
    """
    Fill not-finite gaps by linear interpolation using `scipy.interp1d`with bounds_error
    turned off by default. Keywords to this function are passed to interp1d.
    :param y:
    :return:
    """
    x = np.arange(y.shape[0])
    valid_idx = np.where(np.isfinite(y))[0]
    if len(valid_idx) < 2:
        return y
    f = interp1d(x[valid_idx], y[valid_idx],
                 bounds_error=False,
                 **interp_kwargs
                 )
    return f(x)

def idl_smooth(x: np.ndarray,
               window: int
               ) -> np.ndarray:
    """ Implementation of the IDL smooth(x, window, /EDGE_TRUNCATE, /NAN)"""
    smoothed = np.full(x.shape, np.nan)
    n = len(x)
    for i in np.arange(n):
        kernel_halfsize = np.floor((window-1)/2).astype(int)
        if i < kernel_halfsize:
            kernel_halfsize = i
        if (n-1-i) < kernel_halfsize:
            kernel_halfsize = n-1-i
        smoothed[i] = bn.nanmean(x[i-kernel_halfsize:i+kernel_halfsize+1])
    return smoothed


def fill_nan(y):
    # DEPR
    result = np.copy(y)
    # Get first and last valid index
    no_nan = np.where(np.isfinite(y))[0]
    if len(no_nan) == 0:
        return result
    valid0, valid1 = np.amin(no_nan), np.amax(no_nan)
    # Cut the "inside" section that is bounded by valid measurements
    y_inside = y[valid0:valid1+1]
    x = np.arange(len(y_inside))
    valid_inside = np.where(np.isfinite(y_inside))[0]
    # Interpolate inside range of valid entries
    try:
        func = interp1d(x[valid_inside], y_inside[valid_inside],
                        bounds_error=False)
        interpolated_inside = func(x)
        result[valid0:valid1+1] = interpolated_inside
    except ValueError:
        # May not be applicable (no inner nan ranges)
        pass
    # fill nan-borders with first/last valid value
    if valid0 != 0:
        result[:valid0] = y[valid0]
    if valid1 != len(y)-1:
        result[valid1+1:len(y)+1] = y[valid1]
    return result
