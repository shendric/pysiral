# -*- coding: utf-8 -*-
"""
Created on Sat Aug 01 17:33:02 2015

@author: Stefan

TODO: Evaluate usefulness (or move to internal module)

"""


import contextlib
import numpy as np
from cftime import num2pydate
from netCDF4 import Dataset

from pysiral.core.errorhandler import ErrorStatus
from pysiral.core.output import NCDateNumDef


# TODO: Replace by xarray
class ReadNC(object):
    """
    Quick & dirty method to parse content of netCDF file into a python object
    with attributes from file variables
    """
    def __init__(self, filename, verbose=False, autoscale=True,
                 nan_fill_value=False, global_attrs_only=False):
        self.error = ErrorStatus()
        self.time_def = NCDateNumDef()
        self.keys = []
        self.parameters = []
        self.attributes = []
        self.verbose = verbose
        self.autoscale = autoscale
        self.global_attrs_only = global_attrs_only
        self.nan_fill_value = nan_fill_value
        self.filename = filename
        self.parameters = []
        self.read_globals()
        self.read_content()

    def read_globals(self):
        pass
#        self.gobal_attributes = {}
#        f = Dataset(self.filename)
#        print f.ncattrs()
#        f.close()

    def read_content(self):

        # Open the file
        try:
            f = Dataset(self.filename)
            f.set_auto_scale(self.autoscale)
        except RuntimeError as re:
            raise RuntimeError(f"Cannot read netCDF file: {self.filename}") from re

        # Try to update the time units
        # NOTE: This has become necessary with the use of
        #       variable epochs
        with contextlib.suppress(KeyError, AttributeError):
            time_units = f.variables["time"].units
            self.time_def.units = time_units

        # Get the global attributes
        for attribute_name in f.ncattrs():

            self.attributes.append(attribute_name)
            attribute_value = getattr(f, attribute_name)

            # Convert timestamps back to datetime objects
            # TODO: This needs to be handled better
            if attribute_name in ["start_time", "stop_time"]:
                attribute_value = num2pydate(attribute_value, self.time_def.units,
                                             calendar=self.time_def.calendar)
            setattr(self, attribute_name, attribute_value)

        # Get the variables
        if not self.global_attrs_only:
            for key in f.variables.keys():

                try:
                    variable = f.variables[key][:]
                except ValueError:
                    continue

                try:
                    is_float = variable.dtype in ["float32", "float64"]
                    has_mask = hasattr(variable, "mask")
                except:
                    is_float, has_mask = False, False

                if self.nan_fill_value and has_mask and is_float:
                    is_fill_value = np.where(variable.mask)
                    variable[is_fill_value] = np.nan

                setattr(self, key, variable)
                self.keys.append(key)
                self.parameters.append(key)
                if self.verbose:
                    print(key)
            self.parameters = f.variables.keys()
        f.close()
