# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


from dataclasses import field
from typing import Dict
from pydantic import BaseModel

from pysiral.l1 import L1Algorithm
from pysiral.l1.l1_data import Level1bData
from pysiral.retracker.tfmra import cTFMRA


class TFMRALeadingEdgeWidth(object):
    """
    Container for computation of leading edge width by taking differences
    between first maximum power thresholds
    """

    def __init__(self, rng, wfm, radar_mode, retrack_flag, tfmra_options=None):
        """
        Compute filtered waveform and index of first maximum once. Calling this class
        will cause the a preliminary retracking of the waveforms indicated by the
        retracker flag. The information is stored in the class and the leading edge
        width can be extraced with the `get_width_from_thresholds` method.

        :param rng: (np.array, dim=(n_records, n_range_bins))
            Waveform range bins
        :param wfm: (np.array, dim=(n_records, n_range_bins))
            Waveform power in arbitrary units
        :param radar_mode: (np.array, dim=(n_records))
            radar mode flag
        :param retrack_flag: (np.array, dim=(n_records))
            flag indicating which waveforms should be retracked
        :param tfmra_options: (doct)
        """

        # Init the cTFRMA retracker
        self.tfmra = cTFMRA()
        self.tfmra.set_default_options(tfmra_options)

        # Pre-process the waveforms for retracking and store the result to self
        filt_rng, filt_wfm, fmi, norm = self.tfmra.get_preprocessed_wfm(rng, wfm, radar_mode, retrack_flag)
        self.wfm, self.rng, self.fmi = filt_wfm, filt_rng, fmi

    def get_width_from_thresholds(self, thres0, thres1, **kwargs):
        """
        Returns the range difference in range bin units between two thresholds,
        by subtracting the range value of thresh0 from thresh1. This is done
        for all waveforms passed to this class during initialization.
        Intended to compute the width of the leading edge.
        :param thres0: (float) The minimum threshold
        :param thres1: (float) The minimum threshold
        :return:
        """
        return self.tfmra.get_thresholds_distance(self.rng, self.wfm, self.fmi, thres0, thres1, **kwargs)


class L1PLeadingEdgeWidthConfig(BaseModel):
    """
    Configuration class for the L1PLeadingEdgeWidth pre-processor item
    """
    tfmra_leading_edge_start: float = 0.1
    tfmra_leading_edge_end: float = 0.9
    tfmra_options: Dict = field(default_factory=dict)


class L1PLeadingEdgeWidth(L1Algorithm):
    """
    A L1P pre-processor item class for computing leading edge width of a waveform
    using the TFMRA retracker as the difference between two thresholds. The
    unit for leading edge width are range bins """

    def __init__(self, **cfg):
        """
        Init the class with the mandatory options

        :param cfg: (dict) Required options (see self.required_options)
        """
        self.cfg = L1PLeadingEdgeWidthConfig(**cfg)

    def apply(self, l1: "Level1bData") -> None:
        """
        API class for the Level-1 pre-processor. Functionality is compute leading edge width (full, first half &
        second half) and adding the result to the classifier data group
        :param l1: A Level-1 data instance
        :return: None, Level-1 object is change in place
        """

        # Prepare input
        radar_mode = l1.waveform.radar_mode
        is_valid = l1.surface_type.get_by_name("ocean").flag

        # Compute the leading edge width (requires TFMRA retracking) using
        # -> Use a wrapper for cTFMRA
        width = TFMRALeadingEdgeWidth(l1.waveform.range,
                                      l1.waveform.power,
                                      radar_mode,
                                      is_valid,
                                      tfmra_options=self.cfg.tfmra_options)
        lew = width.get_width_from_thresholds(self.cfg.tfmra_leading_edge_start,
                                              self.cfg.tfmra_leading_edge_end)

        # Add result to classifier group
        l1.classifier.add(lew, "leading_edge_width")

    @property
    def required_options(self):
        return ["tfmra_leading_edge_start",
                "tfmra_leading_edge_end",
                "tfmra_options"]
