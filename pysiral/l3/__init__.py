# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 14:04:27 2015

@author: Stefan
"""
import numpy as np
from loguru import logger
from pysiral import get_cls
from pysiral.core.errorhandler import ErrorStatus
from pysiral.core.output import Level3Output
from pysiral.l2 import L2iNCFileImport
from pysiral.l3.data import L3DataGrid, L2iDataStack

# %% Level 3 Processor


class Level3Processor(object):

    def __init__(self, product_def):
        self.error = ErrorStatus(caller_id=self.__class__.__name__)
        self._job = product_def
        self._l3_progress_percent = 0.0
        self._l2i_files = None
        self._period = None

    def process_l2i_files(self, l2i_files, period):
        """
        The main call for the Level-3 processor
        TODO: Needs organization
        :param l2i_files:
        :param period:
        :return:
        """

        # Store l2i_files
        self._l2i_files = l2i_files

        # Store
        self._period = period

        # Initialize the stack for the l2i orbit files
        logger.info("Initialize l2i data stack")
        stack = L2iDataStack(self._job.grid, self._job.l2_parameter)

        logger.info("Parsing products (prefilter active: %s)" % (str(self._job.l3def.l2i_prefilter.active)))

        # Parse all orbit files and add to the stack
        for i, l2i_file in enumerate(l2i_files):

            self._log_progress(i)

            # Parse l2i source file
            try:
                l2i = L2iNCFileImport(l2i_file)
            except AttributeError:
                logger.warning("Attribute Error encountered in %s" % l2i_file)
                continue

            # Apply the orbit filter (for masking descending or ascending orbit segments)
            # NOTE: This tag may not be present in all level-3 settings files, as it has
            #       been added as a test case
            # TODO: Create a configurable processor item
            orbit_filter = self._job.l3def.get("orbit_filter")
            if orbit_filter is not None:
                self.apply_orbit_filter(l2i, orbit_filter)

            # Apply the orbit filter (for masking descending or ascending orbit segments)
            # NOTE: This tag may not be present in all level-3 settings files, as it has
            #       been added as a test case
            # TODO: Create a configurable processor item
            miz_filter = self._job.l3def.get("miz_filter")
            if miz_filter is not None:
                self.apply_miz_filter(l2i, miz_filter)

            # Prefilter l2i product
            # Note: In the l2i product only the minimum set of nan are used
            #       for different parameters (e.g. the radar freeboard mask
            #       does not equal the thickness mask). This leads to
            #       inconsistent results during gridding and therefore it is
            #       highly recommended to harmonize the mask for thickness
            #       and the different freeboard levels
            # TODO: Create a configurable processor item
            prefilter = self._job.l3def.l2i_prefilter
            if prefilter.active:
                l2i.transfer_nan_mask(prefilter.nan_source, prefilter.nan_targets)
            # Add to stack
            stack.add(l2i)

        # Initialize the data grid
        logger.info("Initialize l3 data grid")
        l3 = L3DataGrid(self._job, stack, period)

        # Apply the processing items
        self._apply_processing_items(l3)

        # Write output(s)
        for output_handler in self._job.outputs:
            output = Level3Output(l3, output_handler)
            logger.info("Write %s product: %s" % (output_handler.id, output.export_filename))

    def _log_progress(self, i):
        """ Concise logging on the progress of l2i stack creation """
        n = len(self._l2i_files)
        progress_percent = float(i + 1) / float(n) * 100.
        current_reminder = np.mod(progress_percent, 10)
        last_reminder = np.mod(self._l3_progress_percent, 10)
        if last_reminder > current_reminder:
            logger.info(
                'Creating l2i orbit stack: %3g%% (%g of %g)'
                % (progress_percent - current_reminder, i + 1, n)
            )

        self._l3_progress_percent = progress_percent

    @staticmethod
    def apply_orbit_filter(l2i, orbit_filter):
        """
        Apply a
        :param l2i:
        :param orbit_filter:
        :return:
        """

        # Display warning if filter is active
        logger.warning("Orbit filter is active [%s]" % str(orbit_filter.mask_orbits))

        # Get indices to filter
        if orbit_filter.mask_orbits == "ascending":
            indices = np.where(np.ediff1d(l2i.latitude) > 0.)[0]
        elif orbit_filter.mask_orbits == "descending":
            indices = np.where(np.ediff1d(l2i.latitude) < 0.)[0]
        else:
            logger.error(
                "Invalid orbit filter target, needs to be [ascending, descending], Skipping filter ...")
            indices = []

        # Filter geophysical parameters only
        targets = l2i.parameter_list
        for non_target in ["longitude", "latitude", "timestamp", "time", "surface_type"]:
            try:
                targets.remove(non_target)
            except ValueError:
                pass
        l2i.mask_variables(indices, targets)

    @staticmethod
    def apply_miz_filter(l2i, miz_filter):
        """
        Flag values based on the miz filter value
        :param l2i:
        :param miz_filter:
        :return:
        """

        flag_miz = getattr(l2i, "flag_miz", None)
        if flag_miz is None:
            return

        idx = np.where(flag_miz >= miz_filter["mask_min_value"])[0]
        l2i.mask_variables(idx, miz_filter["mask_targets"])

    def _apply_processing_items(self, l3grid):
        """
        Sequentially apply the processing items defined in the Level-3 processor definition files
        (listed under root.processing_items)
        :param l3grid:
        :return:
        """

        # Get the post processing options
        processing_items = self._job.l3def.get("processing_items", None)
        if processing_items is None:
            logger.info("No processing items defined")
            return

        # Get the list of post-processing items
        for pitem in processing_items:
            msg = "Apply Level-3 processing item: `%s`" % (pitem["label"])
            logger.info(msg)
            pp_class, err = get_cls(pitem["module_name"], pitem["class_name"], relaxed=False)
            processing_items = pp_class(l3grid, **pitem["options"])
            processing_items.apply()


# %% Data Containers


