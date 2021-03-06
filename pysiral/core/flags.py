# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks"

import numpy as np
from collections import OrderedDict

from pysiral._class_template import DefaultLoggingClass


# The standard ESA surface type flag in L1B data
ESA_SURFACE_TYPE_DICT = {
    "ocean": 0,
    "closed_sea": 1,
    "land_ice": 2,
    "land": 3}


SURFACE_TYPE_DICT = {
    "unknown": 0,
    "ocean": 1,
    "lead": 2,
    "polynya": 3,
    "sea_ice": 4,
    "closed_sea": 5,
    "land_ice": 6,
    "land": 7,
    "invalid": 8}


class SurfaceType(DefaultLoggingClass):
    """
    Container for surface type information.

    Possible classifications (Adapted from CryoSat-2 conventions)
        - unknown
        - ocean
        - closed sea/lakes
        - lead
        - large lead/polynya
        - sea ice (general sea ice class, not to be confused with ice type)
        - continental ice
        - land
    """

    def __init__(self):
        """

        """
        super(SurfaceType, self).__init__(self.__class__.__name__)
        self.surface_type_dict = dict(**SURFACE_TYPE_DICT)
        self._surface_type_flags = []
        self._surface_type = None

    def name(self, flag_value):
        """
        Return the flag name for a give flag value
        :param flag_value:
        :return:
        """
        i = list(self.surface_type_dict.values()).index(flag_value)
        return list(self.surface_type_dict.keys())[i]

    def set_flag(self, flag):
        self._surface_type = flag

    def add_flag(self, flag, type_str):
        """
        Add a flag to the
        :param flag:
        :param type_str:
        :return:
        """

        # Add a surface type flag
        if type_str not in self.surface_type_dict.keys():
            msg = "surface type str %s unknown" % type_str
            self.error.add_error("invalid-surface-type-code", msg)

        if self.invalid_n_records(len(flag)):
            msg = "invalid number of records: %g (must be %g)" % (len(flag), self.n_records)
            self.error.add_error("invalid-variable-length", msg)

        self.error.raise_on_error()

        # Create Flag keyword if necessary
        if self._surface_type is None:
            self._surface_type = np.zeros(shape=len(flag), dtype=np.int8)

        # Update surface type list
        indices = np.where(flag)[0]
        self._surface_type[indices] = self._get_type_id(type_str)
        self._surface_type_flags.append(type_str)

    def has_flag(self, type_str):
        return type_str in self._surface_type_flags

    def get_by_name(self, name):
        if name in self.surface_type_dict.keys():
            type_id = self._get_type_id(name)
            return FlagContainer(self._surface_type == type_id)
        else:
            return FlagContainer(np.zeros(shape=self.n_records, dtype=np.bool))

    def append(self, annex):
        self._surface_type = np.append(self._surface_type, annex.flag)

    def set_subset(self, subset_list):
        self._surface_type = self._surface_type[subset_list]

    def fill_gaps(self, corrected_n_records, indices_map):
        """ API gap filler method. Note: Gaps will be filled with
        the nodata=unkown (8) surface_type"""
        dtype = self.flag.dtype
        surface_type_corrected = np.full(corrected_n_records, 8, dtype=dtype)
        surface_type_corrected[indices_map] = self._surface_type
        self._surface_type = surface_type_corrected

    def invalid_n_records(self, n):
        """ Check if flag array has the correct length """
        if self.n_records is None:  # New flag, ok
            return False
        elif self.n_records == n:   # New flag has correct length
            return False
        else:                       # New flag has wrong length
            return True

    def _get_type_id(self, name):
        return self.surface_type_dict[name]

    @property
    def flag(self):
        return self._surface_type

    @property
    def n_records(self):
        if self._surface_type is None:
            n_records = None
        else:
            n_records = len(self._surface_type)
        return n_records

    @property
    def dimdict(self):
        """ Returns dictionary with dimensions"""
        dimdict = OrderedDict([("n_records", self.n_records)])
        return dimdict

    @property
    def parameter_list(self):
        return ["flag"]

    @property
    def lead(self):
        return self.get_by_name("lead")

    @property
    def sea_ice(self):
        return self.get_by_name("sea_ice")

    @property
    def land(self):
        return self.get_by_name("land")


# TODO: Has this been used yet?
class IceType(object):
    """
    Container for ice type information

    Possible classifications
        - young thin ice
        - first year ice
        - multi year ice
        - wet ice
    """
    _ICE_TYPE_DICT = {
        "thin_ice": 0,
        "first_year_ice": 1,
        "multi_year_ice": 2,
        "wet_ice": 3}

    def __init__(self):
        self._ice_type_flag = None


class FlagContainer(object):

    def __init__(self, flag):
        self._flag = flag

    def set_flag(self, flag):
        self._flag = flag

    @property
    def indices(self):
        return np.where(self._flag)[0]

    @property
    def flag(self):
        return self._flag

    @property
    def num(self):
        return len(self.indices)


class ANDCondition(FlagContainer):

    def __init__(self):
        super(ANDCondition, self).__init__(None)

    def add(self, flag):
        if self._flag is None:
            self.set_flag(flag)
        else:
            self.set_flag(np.logical_and(self.flag, flag))


class ORCondition(FlagContainer):

    def __init__(self):
        super(ORCondition, self).__init__(None)

    def add(self, flag):
        if self._flag is None:
            self.set_flag(flag)
        else:
            self.set_flag(np.logical_or(self.flag, flag))
