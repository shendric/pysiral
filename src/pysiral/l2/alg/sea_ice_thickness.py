# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:15:39 2016

@author: shendric
"""
__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = ["SeaIceFreeboard2SIT",]

import numpy as np

from pysiral.l2 import L2Algorithm


class SeaIceFreeboard2SIT(L2Algorithm):
    """
    Classical freeboard to thickness conversion assuming that the variable freeboard
    represent sea ice freeboard.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, l1, l2):
        """
        Compute sea ice thickness based on a Level-2 data object
        - Assuming freeboard is sea ice freeboard
        :param l1b:
        :param l2:
        :return:
        """

        # Short cuts
        rho_w = self.cfg.options.water_density

        # Compute sea ice thickness ...
        sit = icefreeboard2thickness(l2.frb, l2.sd, rho_w, l2.idens, l2.sdens)

        # ... and sea ice thickness uncertainty
        sit_unc = frb2sit_errprop(l2.frb, l2.sd, rho_w, l2.idens, l2.sdens,
                                  l2.frb.uncertainty, l2.sd.uncertainty,
                                l2.idens.uncertainty, l2.sdens.uncertainty)

        # Add parameter to Level-2 data object
        l2.sit.set_value(sit)
        l2.sit.set_uncertainty(sit_unc)

        # Remove any values for non-sea ice waveforms
        is_not_ice = np.logical_not(l2.surface_type.sea_ice.flag)
        l2.sit.set_nan_indices(np.where(is_not_ice))

        # Return the not sea-ice as error status flag
        return is_not_ice

    @property
    def l2_dependencies(self):
        return ["frb", "sd", "sdens", "idens"]

    @property
    def l2_output_vars(self):
        return ["sit"]

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["sit"]


class SnowFreeboard2SIT(SeaIceFreeboard2SIT):
    """
    Variant of the SeaIceFreeboard2SIT, but for snow freeboard to thickness conversion.
    This is implemented as the children to SeaIceFreeboard2SIT, with only different
    pointer to the correct functions
    """

    def __init__(self, *args, **kwargs):
        super(SnowFreeboard2SIT, self).__init__(*args, **kwargs)

    @property
    def func(self):
        """
        Return the function for transforming sea-ice freeboard to thickness
        :return:
        """
        return snowfreeboard2thickness

    @property
    def uncfunc(self):
        """
        Return the function for computing sea-ice thickness uncertainty
        :return:
        """
        # TODO: Is this correct?
        return frb2sit_errprop


def icefreeboard2thickness(frb, sd, rho_w, rho_i, rho_s):
    return rho_w / (rho_w - rho_i) * frb + rho_s / (rho_w - rho_i) * sd


def snowfreeboard2thickness(frb, sd, rho_w, rho_i, rho_s):
    return rho_w / (rho_w - rho_i) * frb + (rho_s - rho_w) / (rho_w - rho_i) * sd


def frb2sit_errprop(frb, sd, rho_w, rho_i, rho_s, frb_unc, sd_unc, rho_i_unc, rho_s_unc):

    # Compute the partial derivates for the error propagation
    deriv_fb = rho_w / (rho_w - rho_i)
    deriv_rho_s = sd / (rho_w - rho_i)
    deriv_rho_i = (frb * rho_w + sd * rho_s) / (rho_w - rho_i)**2.
    deriv_sd = rho_s / (rho_w - rho_i)

    # Error propagation
    sit_unc = np.sqrt((deriv_fb**2. * frb_unc**2) +
                      (deriv_rho_i**2 * rho_i_unc**2) +
                      (deriv_sd**2. * sd_unc**2.) +
                      (deriv_rho_s**2 * rho_s_unc**2.))

    return sit_unc
