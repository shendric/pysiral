# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


from pysiral.l2 import L2Algorithm


class AlexandrovSeaIceDensity(L2Algorithm):
    """
    A Level-2 processor step that adds sea ice density and sea ice density uncertainty
    to the L2 data object based on sea ice type classification
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, l1, l2):
        """
        Mandatory method for the Level2ProcessorStep class. Will add sea ice density
        and uncertainty computed from the sea ice type (interpreted as multi-year ice fraction)
        and add output parameters to the L2 data object
        :param l1:

        :param l2:
        :return: Error Status
        """

        # Get the mandatory error status output
        error_status = self.get_clean_error_status(l2.n_records)

        # Boundary conditions (all fyi, all myi)
        rho_i_fyi = self.cfg.options.fyi_density
        rho_i_myi = self.cfg.options.myi_density

        # Scales with MYI fraction
        myi_fraction_var_name = self.cfg.options.get("myi_fraction_var_name", "sitype")
        myi_fraction = l2.get_parameter_by_name(myi_fraction_var_name)
        rho_i = rho_i_fyi - myi_fraction * (rho_i_fyi - rho_i_myi)

        # Compute uncertainty of sea ice density
        # Note: it has been decided to use a simplified computation of
        # sea ice density uncertainty, since a simple Gaussian error
        # propagation does not cut it, as the two (fyi & myi) reference
        # uncertainties are not independent (linked via scaling factor).
        # One could add the covariance matrix, but this is hardly justified
        # given the already very basic assumptions for fyi and myi density
        # uncertainty.
        # Here, the uncertainty is based on the fyi and myi density
        # uncertainties scaled with the myi fraction. To reflect the
        # myi_fraction uncertainty, the sea ice density uncertainty is
        # slightly increased in regions of higher myi_fraction uncertainty
        if "uncertainty" in self.cfg.options:
            unc = self.cfg.options.uncertainty
            rho_i_unc = unc.fyi_density
            rho_i_unc += myi_fraction * (unc.myi_density - unc.fyi_density)
            rho_i_unc += (unc.fyi_density - unc.myi_density) * myi_fraction.uncertainty

        # Use fixed value for uncertainty (or 0 if unspecified)
        else:
            unc = 0.0
            if "uncertainty_fixed_value" in self.cfg.options:
                unc = self.cfg.options.uncertainty_fixed_value
            rho_i_unc = np.full(rho_i.shape, unc, dtype=np.float32)

        # Add data to the Level-2 data object
        l2.set_auxiliary_parameter("idens", "sea_ice_density", rho_i, rho_i_unc)

        # Done
        return error_status

    @property
    def l2_dependencies(self):
        return ["sitype"]

    @property
    def l2_output_vars(self):
        return ["idens"]

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["sit"]
