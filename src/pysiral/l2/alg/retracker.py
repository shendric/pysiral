# -*- coding: utf-8 -*-
"""
This module contains the Level-2 retracker adapter to all retrackers in the
`pysiral.retracker` module.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from loguru import logger

import time

from pysiral.l2 import L2Algorithm
from pysiral.retracker import get_retracker_class


class Level2RetrackerContainer(L2Algorithm):
    """
    The interface for the Level-2 processor for all retrackers
    """

    def __init__(self, cfg, **kwargs):
        """
        Initialize the instance
        :param args:
        :param kwargs:
        """
        self.cfg = cfg

    def apply(self, l1, l2):
        """
        Mandatory class
        :param l1:
        :param l2:
        :return:
        """

        # Get the error status
        error_status = self.get_clean_error_status(l2.n_records)

        # Retracker are surface type dependent
        # -> Loop over all requested surface types in the
        #    l2 processor definition file
        for surface_type, retracker_def in list(self.cfg.options.items()):

            # Check first if there are any waveforms of the requested
            # surface type
            surface_type_flag = l2.surface_type.get_by_name(surface_type)
            if surface_type_flag.num == 0:
                logger.info(f"- no waveforms of type {surface_type}")
                continue
            else:
                logger.info(
                    f"Retrack {surface_type_flag.num} {surface_type} waveforms with " + retracker_def["pyclass"]
                )

            # Benchmark retracker performance
            timestamp = time.time()

            # Retrieve the retracker associated with surface type from the l2 settings
            retracker = get_retracker_class(retracker_def["pyclass"])

            # Set options (if any)
            retracker_options = {"surface_type": surface_type}
            if retracker_def["options"] is not None:
                retracker_options.update(retracker_def["options"])
            retracker.set_options(**retracker_options)

            # set subset of waveforms
            retracker.set_indices(surface_type_flag.indices)

            # Add classifier data (some retracker need that)
            retracker.set_classifier(l1.classifier)

            # Start the retracker procedure
            retracker.retrack(l1, l2)

            # Update range value of the Level-2 data object
            l2.update_retracked_range(retracker)

            # XXX: Let the retracker return other parameters?
            l2.radar_mode = l1.waveform.radar_mode

            # retrieve potential error status and update surface type flag
            if retracker.error_flag.num > 0:
                l2.surface_type.add_flag(retracker.error_flag.flag, "invalid")
            logger.info("- Retrack class %s with %s in %.3f seconds" % (
                surface_type, retracker_def["pyclass"],
                time.time()-timestamp))

        return error_status

    @property
    def l2_dependencies(self):
        return []

    @property
    def l2_output_vars(self):
        return ["radar_mode", "range", "elev"]

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["retracker"]