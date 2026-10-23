# -*- coding: utf-8 -*-
"""
This module contains Level-1 Pre-Processor algorithms specifically
for computing the radar backscatter coefficient (Sigma0)
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from loguru import logger

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from pysiral.l1 import L1Algorithm
from pysiral.l1.alg import L1PProcItem


def get_waveforms_peak_power(wfm: npt.NDArray, use_db: bool = False) -> npt.NDArray:
    """
    Return the peak power (in input coordinates) of an array of waveforms

    Arguments
    ---------
        wfm (float array)
            echo waveforms, order = (n_records, n_range_bins)

    Returns
    -------
        float array with maximum for each echo
    """
    peak_power = np.amax(wfm, axis=1)
    if use_db:
        peak_power = 10 * np.log10(peak_power)
    return peak_power


def get_footprint_pulse_limited(r: float, band_width: float) -> float:
    """
    Compute the CryoSat-2 LRM footprint for variable range to the surface.

    Applicable Documents:

        Michele Scagliola, CryoSat Footprints ESA/Aresys, v1.2
        ESA document ref: XCRY-GSEG-EOPG-TN-13-0013

    :param r: range from satellite center of mass to surface reflection point
    :param band_width: pulse bandwidth in Hz

    :return area_pl: Radar backscatter coefficient
    """

    c_0 = 299792458.0
    footprint_radius = np.sqrt(r * c_0 / band_width) / 1000.
    return np.pi * footprint_radius ** 2.


def get_footprint_sar(r: float,
                      v_s: float,
                      ptr_width: float,
                      tau_b: float,
                      lambda_0: float,
                      wf: float = 1.0,
                      r_mean: float = 6371000.0
                      ) -> float:
    """
    Compute the radar footprint of SAR altimeters for variable range to the surface.

    Applicable Documents:

        Michele Scagliola, CryoSat Footprints ESA/Aresys, v1.2
        ESA document ref: XCRY-GSEG-EOPG-TN-13-0013

    :param r: range from satellite center of mass to surface reflection point
    :param v_s: satellite along track velocity in meter/sec
    :param ptr_width: 3dB range point target response temporal width in seconds
    :param tau_b: burst length in seconds
    :param lambda_0: radar wavelength in meter
    :param wf: footprint widening factor
    :param r_mean: mean earth radius in meter

    :return area_sar: The SAR footprint in square meters
    """
    c_0 = 299792458.0
    alpha_earth = 1. + (r / r_mean)
    lx = (lambda_0 * r) / (2. * v_s * tau_b)
    ly = np.sqrt((c_0 * r * ptr_width) / alpha_earth)
    return (2. * ly) * (wf * lx)


def get_sigma0_sar(rx_pwr: float,
                   tx_pwr: float,
                   r: float,
                   a: float,
                   lambda_0: float,
                   g_0: float,
                   l_atm: float = 1.0,
                   l_rx: float = 1.0,
                   bias_sigma0: float = 0.0,
                   ) -> float:
    """
    Compute the sigma0 backscatter coefficient according the radar equation, e.g.
    equation 20 in

        Guidelines for reverting Waveform Power to Sigma Nought for
        CryoSat-2 in SAR mode (v2.2), Salvatore Dinardo, 23/06/2016
        XCRY-GSEG-EOPS-TN-14-0012

    :param rx_pwr: received power (waveform maximumn)
    :param tx_pwr: transmitted power
    :param r: range to surface in ms
    :param a: area illuminated by the altimeter
    :param lambda_0: radar wavelength in meter
    :param g_0: antenna gain factor
    :param l_atm: atmospheric loss factor (1.0 -> no loss)
    :param l_rx: receiving chain losses
    :param bias_sigma0: sigma0 bias in dB
    :return: sigma0 in dB
    """

    k = ((4.*np.pi)**3. * r**4. * l_atm * l_rx)/(lambda_0**2. * g_0**2. * a)
    return 10. * np.log10(rx_pwr/tx_pwr) + 10. * np.log10(k) + bias_sigma0


def get_sigma0(wf_peak_power_watt: float,
               tx_pwr: float,
               r: float,
               wf_thermal_noise_watt: float = 0.0,
               lambda_0: float = 0.022084,
               band_width: float = 320000000.0,
               g_0: float = 19054.607179632483,
               bias_sigma0: float = 0.0,
               l_atm: float = 1.0,
               l_rx: float = 1.0,
               c_0: float = 299792458.0,
               ) -> float:
    """
    Compute the radar backscatter coefficient sigma nought (sigma0) for lrm waveforms. Uses the radar
    equation and the computation of the pulse limited footprint

    Applicable Documents:

        Guidelines for reverting Waveform Power to Sigma Nought for
        CryoSat-2 in SAR mode (v2.2), Salvatore Dinardo, 23/06/2016
        XCRY-GSEG-EOPS-TN-14-0012

    :param wf_peak_power_watt:  waveform peak power in watt
    :param tx_pwr: transmitted peak power in watt
    :param r: range from satellite center of mass to surface reflection point
        (to be appoximated by satellite altitude if no retracker range available)
    :param wf_thermal_noise_watt: estimate of thermal noise power in watt (default: 0.0)
        will be used to estimate waveform amplitude (Pu)
    :param band_width: CryoSat-2 pulse bandwidth in Hz
    :param lambda_0: radar wavelength in meter (default: 0.022084 m for CryoSat-2 Ku Band altimeter)
    :param g_0: antenna gain at boresight (default: 10^(4.28) from document)
    :param bias_sigma0: sigma nought bias (default: 0.0)
    :param l_atm: two ways atmosphere losses (to be modelled) (default: 1.0 (no loss))
    :param l_rx: receiving chain (RX) waveguide losses (to be characterized) (default: 1.0 (no loss))
    :param c_0: vacuum light speed in meter/sec

    :return sigma_0: Radar backscatter coefficient
    """

    # In the document it is referred to as "waveform power value in output
    # of the re-tracking stage", however generally it is referred to as
    # "waveform amplitude" that is obtained by a waveform function fit
    # It is the scope of this function to provide a sigma0 estimate without
    # proper retracking, therefore Pu is simply defined by the peak power
    # and the thermal noise_power in watt
    pu = wf_peak_power_watt + wf_thermal_noise_watt

    # Intermediate steps & variables
    footprint_radius = np.sqrt(r * c_0 / band_width) / 1000.
    a_lrm = np.pi * footprint_radius ** 2.
    k = ((4.*np.pi)**3. * r**4. * l_atm * l_rx)/(lambda_0**2. * g_0**2. * a_lrm)

    return 10. * np.log10(pu/tx_pwr) + 10. * np.log10(k) + bias_sigma0


class L1PSigma0(L1PProcItem):
    """
    A L1P pre-processor item class for computing the backscatter coefficient (sigma0) from
    waveform data

    """

    def __init__(self, **cfg):
        super(L1PSigma0, self).__init__(**cfg)

    def apply(self, l1):
        """
        API class for the Level-1 pre-processor. Functionality is to compute
        leading edge width (full, first half & second half) and adding the result to
        the classifier data group

        :param l1: A Level-1 data instance

        :return: None, Level-1 object is change in place
        """

        # Get input power

        radar_mode = l1.get_parameter_by_name("waveform", "radar_mode")
        rx_power = get_waveforms_peak_power(l1.waveform.power)
        ocog_amplitude = l1.get_parameter_by_name("classifier", "ocog_amplitude")
        is_lrm = radar_mode == 0
        rx_power[is_lrm] = ocog_amplitude[is_lrm]

        # Get Input parameter from l1 object
        tx_power = l1.get_parameter_by_name("classifier", "transmit_power")
        if tx_power is None:
            msg = "classifier `transmit_power` must exist for this pre-processor item -> aborting"
            logger.warning(msg)
            return

        # The computation of sigma0 requires the range to the surface as input
        # and is a step after the retracker. Here we use the satellite altitude
        # as an approximation for sea ice and ocean surfaces.
        altitude = l1.time_orbit.altitude

        # Compute absolute satellite velocity
        sat_vel_x = l1.get_parameter_by_name("classifier", "satellite_velocity_x")
        sat_vel_y = l1.get_parameter_by_name("classifier", "satellite_velocity_y")
        sat_vel_z = l1.get_parameter_by_name("classifier", "satellite_velocity_z")
        if sat_vel_x is None or sat_vel_y is None or sat_vel_z is None:
            # TODO: This is only strictly true for SAR waveforms, observe if necessary
            msg = "classifier `satellite_velocity_[x|y|z]` must exist for this pre-processor item -> aborting"
            logger.warning(msg)
            return
        velocity = np.sqrt(sat_vel_x**2. + sat_vel_y**2. + sat_vel_z**2.)

        # Compute sigma_0
        sigma0 = self.get_sigma0(rx_power, tx_power, altitude, velocity, radar_mode)

        # Add the classifier
        l1.classifier.add(rx_power, "peak_power")
        l1.classifier.add(sigma0, "sigma0")

    def get_sigma0(self,
                   rx_power: np.ndarray,
                   tx_power: np.ndarray,
                   altitude: np.ndarray,
                   velocity: np.ndarray,
                   radar_mode: np.ndarray
                   ) -> np.ndarray:
        """

        :param rx_power:
        :param tx_power:
        :param altitude:
        :param velocity:
        :param radar_mode:
        :return:
        """

        # The function to compute the footprint and the required keywords
        # depend on the radar mode id.
        footprint_func_dict = {0: get_footprint_pulse_limited,
                               1: get_footprint_sar,
                               2: get_footprint_sar}

        footprint_func_kwargs = {0: self.cfg["footprint_pl_kwargs"],
                                 1: self.cfg["footprint_sar_kwargs"],
                                 2: self.cfg["footprint_sar_kwargs"]}

        # The computation of sigma0 depends on properties
        # of the radar altimeter
        sigma0_kwargs = self.cfg.get("sigma0_kwargs", None)

        # Init the output array
        sigma0 = np.full(rx_power.shape, np.nan)

        # Compute sigma0 per waveform
        for i in np.arange(rx_power.shape[0]):

            # Compute the footprint area
            args = (altitude[i], ) if radar_mode[i] == 0 else (altitude[i], velocity[i])
            func = footprint_func_dict[radar_mode[i]]
            footprint_area = func(*args, **footprint_func_kwargs[radar_mode[i]])

            # Compute the backscatter coefficient
            sigma0[i] = get_sigma0_sar(rx_power[i],
                                       tx_power[i],
                                       altitude[i],
                                       footprint_area,
                                       **sigma0_kwargs)

        # Eliminate infinite values
        sigma0[np.isinf(sigma0)] = np.nan

        return sigma0

    @property
    def required_options(self):
        return ["footprint_pl_kwargs", "footprint_sar_kwargs", "sigma0_kwargs", "sigma0_bias"]


class Sigma0FromScalingFactorConfig(BaseModel):
    """
    Configuration class for the L1PSigma0FromScalingFactor pre-processor item
    """
    peak_power_fraction: float = 1.0  # Fraction of the peak power to be used for sigma0 computation
    classifier_target: str  # classifier parameter name to be used for sigma0 computation


class L1PSigma0FromScalingFactor(L1Algorithm):

    def __init__(self, **kwargs):
        """
        Initialize the L1PSigma0FromScalingFactor algorithm with configuration parameters.

        :param kwargs: Configuration parameters for the algorithm
        """
        self.cfg = Sigma0FromScalingFactorConfig(**kwargs)

    def apply(self, l1: "Level1bData") -> None:
        """
        Apply the L1PSigma0FromScalingFactor algorithm to the Level1bData.

        :param l1: Level1bData instance to process
        """
        scaling_factor = l1.get_parameter_by_name("classifier", self.cfg.classifier_target)
        assert scaling_factor is not None, f"Classifier parameter '{self.cfg.classifier_target}' does not exist"
        sigma0 = self.cfg.peak_power_fraction * scaling_factor
        l1.classifier.add(sigma0, "sigma0")