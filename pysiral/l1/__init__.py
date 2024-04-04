# -*- coding: utf-8 -*-

"""
List of ToDos:
:TODO: Add option to log activities and also exceptions in certain directory
:TODO: Add output catalog
"""

from datetime import timedelta
from operator import attrgetter
from pathlib import Path
from typing import Dict, List, Tuple, Literal, Union, Optional, Any

import numpy as np
from collections import defaultdict
from dateperiods import DatePeriod
from loguru import logger

from pysiral import psrlcfg
from pysiral.core.config import DataVersion
from pysiral.core.dataset_ids import SourceDataID
from pysiral.core.clocks import debug_timer
from pysiral.core.helper import ProgressIndicator, get_first_array_index, get_last_array_index, rle
from pysiral.core.output import L1bDataNC
from pysiral.l1.data import L1bMetaData, Level1bData
from pysiral.l1.alg import L1PProcItem, L1PProcItemDef
from pysiral.l1.io import SourceDataLoader, SourceFileDiscovery
from pysiral.l1.cfg import L1pProcessorConfig, PolarOceanSegmentsConfig


from pysiral.l1.debug import l1p_debug_map

__all__ = [
    "Level1POutputHandler", "Level1PreProcessor", "L1pProcessorConfig",
    "Level1bData", "L1bMetaData", "L1PProcItem", "SourceDataLoader",
    "SourceFileDiscovery"
]


class Level1POutputHandler(object):
    """
    """

    def __init__(
            self,
            platform: str,
            timeliness: str,
            l1p_version: str,
            file_version: str
    ) -> None:
        self.platform = platform
        self.timeliness = timeliness
        self.l1p_version = l1p_version
        self.file_version = file_version
        self._last_written_file = None

    @staticmethod
    def remove_old_if_applicable(period: DatePeriod) -> None:
        logger.warning("Not implemented: self.remove_old_if_applicable")
        return

    def export_to_netcdf(self, l1: "Level1bData") -> None:
        """
        Workflow to export a Level-1 object to l1p netCDF product. The workflow includes the generation of the
        output path (if applicable).

        :param l1: The Level-1 object to be exported

        :return: None
        """

        # Get filename and path
        output_directory = self.get_output_directory(l1)
        filename = self.get_output_filename(l1)

        # Check if path exists
        output_directory.mkdir(exist_ok=True, parents=True)

        # Export the data object
        ncfile = L1bDataNC()
        ncfile.l1b = l1
        ncfile.output_folder = output_directory
        ncfile.filename = filename
        ncfile.export()

    def get_output_filename(self, l1: "Level1bData") -> str:
        """
        Construct the filename from the Level-1 data object

        :param l1:

        :return: filename
        """

        filename_template = "pysiral-l1p-{platform}-{source}-{timeliness}-{hemisphere}-{tcs}-{tce}-{file_version}.nc"
        time_fmt = "%Y%m%dT%H%M%S"
        values = {"platform": l1.info.mission,
                  "source": self.l1p_version,
                  "timeliness": l1.info.timeliness,
                  "hemisphere": l1.info.hemisphere,
                  "tcs": l1.time_orbit.timestamp[0].strftime(time_fmt),
                  "tce": l1.time_orbit.timestamp[-1].strftime(time_fmt),
                  "file_version": self.file_version}
        return filename_template.format(**values)

    def get_output_directory(self, l1: "Level1bData") -> Path:
        """
        Sets the class properties required for the file export

        :param l1: The Level-1 object

        :return: None
        """
        export_folder = psrlcfg.local_path.pysiral_output.base_directory
        yyyy = "%04g" % l1.time_orbit.timestamp[0].year
        mm = "%02g" % l1.time_orbit.timestamp[0].month
        return export_folder / "l1p" / self.platform / self.l1p_version / self.file_version / l1.info.hemisphere / yyyy / mm

    @property
    def last_written_file(self) -> Path:
        return self.last_written_file


class PolarOceanSegments(object):
    """
    Class to extract polar ocean segments

    :param orbit_coverage:
    :param target_hemisphere:
    :param polar_latitude_threshold:
    :param allow_nonocean_segment_nrecords:
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize class instance
        """
        self.cfg = PolarOceanSegmentsConfig(**kwargs)

    def extract_polar_ocean_segments_custom_orbit_segment(self, l1: "Level1bData") -> List["Level1bData"]:
        """
        Splits the input Level-1 object into the polar ocean segments (e.g. by trimming land at the edges
        or by splitting into several parts if there are land masses with the orbit segment). The returned
        polar ocean segments should be generally free of data over non-ocean parts of the orbit, except
        for smaller parts within the orbit.

        NOTE: This subclass of the Level-1 Pre-Processor is designed for input data type with arbitrary
              orbit segment length (e.g. data of CryoSat-2 where the orbit segments of the input data
              is controlled by the mode mask changes).

        :param l1: A Level-1 data object

        :return: A list of Level-1 data objects (subsets of polar ocean segments from input l1)
        """

        # Step: Filter small ocean segments
        # NOTE: The objective is to remove any small marine regions (e.g. in fjords) that do not have any
        #       reasonable chance of freeboard/ssh retrieval early on in the pre-processing.
        if "ocean_mininum_size_nrecords" in self.cfg:
            logger.info("- filter ocean segments")
            l1 = self.filter_small_ocean_segments(l1)

        # Step: Trim the orbit segment to latitude range for the specific hemisphere
        # NOTE: There is currently no case of an input data set that is not of type half-orbit and that
        #       would have coverage in polar regions of both hemisphere. Therefore, `l1_subset` is assumed to
        #       be a single Level-1 object instance and not a list of instances.  This needs to be changed if
        #      `input_file_is_single_hemisphere=False`
        logger.info("- extracting polar region subset(s)")
        if l1.is_single_hemisphere:
            l1_list = [self.trim_single_hemisphere_segment_to_polar_region(l1)]
        else:
            l1_list = self.trim_two_hemisphere_segment_to_polar_regions(l1)
        logger.info(f"- extracted {len(l1_list)} polar region subset(s)")

        # Step: Split the l1 segments at time discontinuities.
        # NOTE: This step is optional. It requires the presence of the options `timestamp_discontinuities`
        #       in the L1 pre-processor config file
        if self.cfg.timestamp_discontinuities is not None:
            logger.info("- split at time discontinuities")
            l1_list = self.split_at_time_discontinuities(l1_list)

        # Step: Trim the non-ocean parts of the subset (e.g. land, land-ice, ...)
        # NOTE: Generally it can be assumed that the l1 object passed to this method contains polar ocean data.
        #       But there tests before only include if there is ocean data and data above the polar latitude
        #       threshold. It can therefore happen that trimming the non-ocean data leaves an empty Level-1 object.
        #       In this case an empty list is returned.
        logger.info("- trim outer non-ocean regions")
        l1_trimmed_list = []
        for l1 in l1_list:
            l1_trimmed = self.trim_non_ocean_data(l1)
            if l1_trimmed is not None:
                l1_trimmed_list.append(l1_trimmed)

        # Step: Split the remaining subset at non-ocean parts.
        # NOTE: There is no need to split the orbit at small features. See option `allow_nonocean_segment_nrecords`
        #       in the l1p processor definition. But even if there are no segments to split, the output will always
        #       be a list per requirements of the Level-1 pre-processor workflow.
        l1_list = []
        for l1 in l1_trimmed_list:
            l1_splitted_list = self.split_at_large_non_ocean_segments(l1)
            l1_list.extend(l1_splitted_list)

        # All done, return the list of polar ocean segments
        return l1_list

    def extract_polar_ocean_segments_half_orbit(self, l1: "Level1bData") -> List["Level1bData"]:
        """
        Splits the input Level-1 object into the polar ocean segments (e.g. by trimming land at the edges
        or by splitting into several parts if there are land masses with the orbit segment). The returned
        polar ocean segments should be generally free of data over non-ocean parts of the orbit, except
        for smaller parts within the orbit.

        NOTE: This subclass of the Level-1 Pre-Processor is designed for input data type with coverage
              from pole to pole (e.g. Envisat SGDR)

        :param l1: A Level-1 data object

        :return: A list of Level-1 data objects (subsets of polar ocean segments from input l1)
        """

        # Step: Filter small ocean segments
        # NOTE: The objective is to remove any small marine regions (e.g. in fjords) that do not have any
        #       reasonable chance of freeboard/ssh retrieval early on in the pre-processing.
        if "ocean_mininum_size_nrecords" in self.cfg.polar_ocean:
            logger.info("- filter ocean segments")
            l1 = self.filter_small_ocean_segments(l1)

        # Step: Extract Polar ocean segments from full orbit respecting the selected target hemisphere
        l1_list = self.trim_two_hemisphere_segment_to_polar_regions(l1)
        logger.info(f"- extracted {len(l1_list)} polar region subset(s)")

        # Step: Split the l1 segments at time discontinuities.
        # NOTE: This step is optional. It requires the presence of the options branch `timestamp_discontinuities`
        #       in the L1 pre-processor config file
        if "timestamp_discontinuities" in self.cfg:
            logger.info("- split at time discontinuities")
            l1_list = self.split_at_time_discontinuities(l1_list)

        # Step: Trim the non-ocean parts of the subset (e.g. land, land-ice, ...)
        # NOTE: Generally it can be assumed that the l1 object passed to this method contains polar ocean data.
        #       But there tests before only include if there is ocean data and data above the polar latitude
        #       threshold. It can therefore happen that trimming the non-ocean data leaves an empty Level-1 object.
        #       In this case an empty list is returned.
        logger.info("- trim outer non-ocean regions")
        l1_trimmed_list = []
        for l1 in l1_list:
            l1_trimmed = self.trim_non_ocean_data(l1)
            if l1_trimmed is not None:
                l1_trimmed_list.append(l1_trimmed)

        # Step: Split the remaining subset at non-ocean parts.
        # NOTE: There is no need to split the orbit at small features. See option `allow_nonocean_segment_nrecords`
        #       in the l1p processor definition. But even if there are no segments to split, the output will always
        #       be a list per requirements of the Level-1 pre-processor workflow.
        l1_list = []
        for l1 in l1_trimmed_list:
            l1_splitted_list = self.split_at_large_non_ocean_segments(l1)
            l1_list.extend(l1_splitted_list)

        # All done, return the list of polar ocean segments
        return l1_list

    def extract_polar_ocean_segments_full_orbit(self, l1: "Level1bData") -> List["Level1bData"]:
        """
        Splits the input Level-1 object into the polar ocean segments (e.g. by trimming land at the edges
        or by splitting into several parts if there are land masses with the orbit segment). The returned
        polar ocean segments should be generally free of data over non-ocean parts of the orbit, except
        for smaller parts within the orbit.

        NOTE: This subclass of the Level-1 Pre-Processor is designed for input data type with arbitrary
              orbit segment length (e.g. data of CryoSat-2 where the orbit segments of the input data
              is controlled by the mode mask changes).

        :param l1: A Level-1 data object
        :return: A list of Level-1 data objects (subsets of polar ocean segments from input l1)
        """

        # Step: Filter small ocean segments
        # NOTE: The objective is to remove any small marine regions (e.g. in fjords) that do not have any
        #       reasonable chance of freeboard/ssh retrieval early on in the pre-processing.
        if "ocean_mininum_size_nrecords" in self.cfg:
            logger.info("- filter ocean segments")
            l1 = self.filter_small_ocean_segments(l1)

        # Step: Extract Polar ocean segments from full orbit respecting the selected target hemisphere
        logger.info("- extracting polar region subset(s)")
        l1_list = self.trim_multiple_hemisphere_segment_to_polar_regions(l1)
        logger.info(f"- extracted {len(l1_list)} polar region subset(s)")

        # Step: Split the l1 segments at time discontinuities.
        # NOTE: This step is optional. It requires the presence of the options branch `timestamp_discontinuities`
        #       in the L1 pre-processor config file
        if "timestamp_discontinuities" in self.cfg:
            l1_list = self.split_at_time_discontinuities(l1_list)
            logger.info(f"- split at time discontinuities -> {len(l1_list)} segments")

        # Step: Trim the non-ocean parts of the subset (e.g. land, land-ice, ...)
        # NOTE: Generally it can be assumed that the l1 object passed to this method contains polar ocean data.
        #       But there tests before only include if there is ocean data and data above the polar latitude
        #       threshold. It can therefore happen that trimming the non-ocean data leaves an empty Level-1 object.
        #       In this case an empty list is returned.
        logger.info("- trim outer non-ocean regions")
        l1_trimmed_list = []
        for l1 in l1_list:
            l1_trimmed = self.trim_non_ocean_data(l1)
            if l1_trimmed is not None:
                l1_trimmed_list.append(l1_trimmed)

        # Step: Split the remaining subset at non-ocean parts.
        # NOTE: There is no need to split the orbit at small features. See option `allow_nonocean_segment_nrecords`
        #       in the l1p processor definition. But even if there are no segments to split, the output will always
        #       be a list per requirements of the Level-1 pre-processor workflow.
        l1_list = []
        for l1 in l1_trimmed_list:
            l1_splitted_list = self.split_at_large_non_ocean_segments(l1)
            l1_list.extend(l1_splitted_list)

        # All done, return the list of polar ocean segments
        return l1_list

    def extract(self, l1: Level1bData) -> Optional[List[Level1bData]]:
        """
        Extract polar ocean segments from a Level-1 data object.

        :param l1: Level-1 data object

        :return: List of Level-1 data objects with only polar ocean data
        """
        method_dict = {
            "custom_orbit_segment": self.extract_polar_ocean_segments_custom_orbit_segment,
            "half_orbit": self.extract_polar_ocean_segments_half_orbit,
            "full_orbit": self.extract_polar_ocean_segments_full_orbit
        }
        return method_dict[self.cfg.orbit_coverage](l1)

    def check(self, product_metadata: L1bMetaData) -> bool:
        """
        Checks if there are polar oceans segments based on the metadata of a L1 data object

        :param product_metadata: Metadata container of the l1b data product.

        :return: Boolean Flag (true: in region of interest, false: not in region of interest)
        """

        # 1 Check: Needs ocean data
        if product_metadata.open_ocean_percent <= 1e-6:
            logger.info("- No ocean data")
            return False

        # 2. Must be in target hemisphere
        # NOTE: the definition of hemisphere in l1 data is above or below the equator
        hemisphere = product_metadata.hemisphere
        target_hemisphere = self.cfg.get("target_hemisphere", None)
        if hemisphere != "global" and hemisphere not in target_hemisphere:
            logger.info(f'- No data in target hemishere: {"".join(self.cfg.target_hemisphere)}')
            return False

        # 3. Must be at higher latitude than the polar latitude threshold
        lat_range = np.abs([product_metadata.lat_min, product_metadata.lat_max])
        polar_latitude_threshold = self.cfg.get("polar_latitude_threshold", None)
        if np.amax(lat_range) < polar_latitude_threshold:
            msg = "- No data above polar latitude threshold (min:%.1f, max:%.1f) [req:+/-%.1f]"
            msg %= (product_metadata.lat_min, product_metadata.lat_max, polar_latitude_threshold)
            logger.info(msg)
            return False

        # 4. All tests passed
        return True

    def trim_single_hemisphere_segment_to_polar_region(self, l1: Level1bData) -> Level1bData:
        """
        Extract polar region of interest from a segment that is either north or south (not global)

        :param l1: Input Level-1 object

        :return: Trimmed Input Level-1 object
        """
        is_polar = np.abs(l1.time_orbit.latitude) >= self.cfg.polar_latitude_threshold
        polar_subset = np.where(is_polar)[0]
        if len(polar_subset) != l1.n_records:
            l1.trim_to_subset(polar_subset)
        return l1

    def trim_two_hemisphere_segment_to_polar_regions(
            self, l1: Level1bData
    ) -> Union[None, List[Level1bData]]:
        """
        Extract polar regions of interest from a segment that is either north, south or both. The method will
        preserve the order of the hemispheres

        :param l1: Input Level-1 object
        :return: List of Trimmed Input Level-1 objects
        """

        polar_threshold = self.cfg.polar_latitude_threshold
        l1_list = []

        # Loop over the two hemispheres
        for hemisphere in self.cfg.target_hemisphere:

            if hemisphere == "north":
                is_polar = l1.time_orbit.latitude >= polar_threshold

            elif hemisphere == "south":
                is_polar = l1.time_orbit.latitude <= (-1.0 * polar_threshold)

            else:
                raise ValueError(f"Unknown {hemisphere=} [north|south]")

            # Extract the subset (if applicable)
            polar_subset = np.where(is_polar)[0]
            n_records_subset = len(polar_subset)

            # is true subset -> add subset to output list
            if n_records_subset != l1.n_records and n_records_subset > 0:
                l1_segment = l1.extract_subset(polar_subset)
                l1_list.append(l1_segment)

            # entire segment in polar region -> add full segment to output list
            elif n_records_subset == l1.n_records:
                l1_list.append(l1)

        # Last step: Sort the list to maintain temporal order
        # (only if more than 1 segment)
        if len(l1_list) > 1:
            l1_list = sorted(l1_list, key=attrgetter("tcs"))

        return l1_list

    def trim_multiple_hemisphere_segment_to_polar_regions(
            self, l1: Level1bData
    ) -> Union[None, List[Level1bData]]:
        """
        Extract polar regions segments from an orbit segment that may cross from north to south
        to north again (or vice versa).

        :param l1: Input Level-1 object

        :return: List of Trimmed Input Level-1 objects
        """

        # Compute flag for segments in polar regions
        # regardless of hemisphere
        polar_threshold = self.cfg.polar_latitude_threshold
        is_polar = np.array(np.abs(l1.time_orbit.latitude) >= polar_threshold)

        # Find start and end indices of continuous polar
        # segments based on the change of the `is_polar` flag
        change_to_polar = np.ediff1d(is_polar.astype(int))
        change_to_polar = np.insert(change_to_polar, 0, 1 if is_polar[0] else 0)
        change_to_polar[-1] = -1 if is_polar[-1] else change_to_polar[-1]
        start_idx = np.where(change_to_polar > 0)[0]
        end_idx = np.where(change_to_polar < 0)[0]

        # Create a list of l1 subsets
        l1_list = []
        for i in np.arange(len(start_idx)):
            polar_idxs = np.arange(start_idx[i], end_idx[i])
            l1_segment = l1.extract_subset(polar_idxs)
            l1_list.append(l1_segment)

        return l1_list

    def trim_full_orbit_segment_to_polar_regions(
            self, l1: Level1bData
    ) -> Union[None, List[Level1bData]]:
        """
        Extract polar regions of interest from a segment that is either north, south or both. The method will
        preserve the order of the hemispheres

        :param l1: Input Level-1 object
        :return: List of Trimmed Input Level-1 objects
        """

        polar_threshold = self.cfg.polar_latitude_threshold
        l1_list = []

        # Loop over the two hemispheres
        for hemisphere in self.cfg.target_hemisphere:

            # Compute full polar subset range
            if hemisphere == "nh":
                is_polar = l1.time_orbit.latitude >= polar_threshold
            elif hemisphere == "sh":
                is_polar = l1.time_orbit.latitude <= (-1.0 * polar_threshold)
            else:
                raise ValueError(f"Unknown {hemisphere=} [nh|sh]")

            # Step: Extract the polar ocean segment for the given hemisphere
            polar_subset = np.where(is_polar)[0]
            n_records_subset = len(polar_subset)

            # Safety check
            if n_records_subset == 0:
                continue
            l1_segment = l1.extract_subset(polar_subset)

            # Step: Trim non-ocean segments
            l1_segment = self.trim_non_ocean_data(l1_segment)

            # Step: Split the polar subset to its marine regions
            l1_segment_list = self.split_at_large_non_ocean_segments(l1_segment)

            # Step: append the ocean segments
            l1_list.extend(l1_segment_list)

        # Last step: Sort the list to maintain temporal order
        # (only if more than 1 segment)
        if len(l1_list) > 1:
            l1_list = sorted(l1_list, key=attrgetter("tcs"))

        return l1_list

    def filter_small_ocean_segments(self, l1: "Level1bData") -> "Level1bData":
        """
        This method sets the surface type flag of very small ocean segments to land.
        This action should prevent large portions of land staying in the l1 segment
        is a small fjord et cetera is crossed. It should also filter out smaller
        ocean segments that do not have a realistic chance of freeboard retrieval.

        :param l1: A pysiral.l1bdata.Level1bData instance

        :return: filtered l1 object
        """

        # Minimum size for valid ocean segments
        ocean_mininum_size_nrecords = self.cfg.polar_ocean.ocean_mininum_size_nrecords

        # Get the clusters of ocean parts in the l1 object
        ocean_flag = l1.surface_type.get_by_name("ocean").flag
        land_flag = l1.surface_type.get_by_name("land").flag
        segments_len, segments_start, not_ocean = rle(ocean_flag)

        # Find smaller than threshold ocean segments
        small_cluster_indices = np.where(segments_len < ocean_mininum_size_nrecords)[0]

        # Do not mess with the l1 object if not necessary
        if len(small_cluster_indices) == 0:
            return l1

        # Set land flag -> True for small ocean segments
        for small_cluster_index in small_cluster_indices:
            i0 = segments_start[small_cluster_index]
            i1 = i0 + segments_len[small_cluster_index]
            land_flag[i0:i1] = True

        # Update the l1 surface type flag by re-setting the land flag
        l1.surface_type.add_flag(land_flag, "land")

        # All done
        return l1

    @staticmethod
    def trim_non_ocean_data(l1: "Level1bData") -> Union[None, "Level1bData"]:
        """
        Remove leading and trailing data that is not if type ocean.

        :param l1: The input Level-1 objects

        :return: The subsetted Level-1 objects. (Segments with no ocean data are removed from the list)
        """

        ocean = l1.surface_type.get_by_name("ocean")
        first_ocean_index = get_first_array_index(ocean.flag, True)
        last_ocean_index = get_last_array_index(ocean.flag, True)
        if first_ocean_index is None or last_ocean_index is None:
            return None
        n = l1.info.n_records - 1
        is_full_ocean = first_ocean_index == 0 and last_ocean_index == n
        if not is_full_ocean:
            ocean_subset = np.arange(first_ocean_index, last_ocean_index + 1)
            l1.trim_to_subset(ocean_subset)
        return l1

    def split_at_large_non_ocean_segments(self, l1: "Level1bData") -> List["Level1bData"]:
        """
        Identify larger segments that are not ocean (land, land ice) and split the segments if necessary.
        The return value will always be a list of Level-1 object instances, even if no non-ocean data
        segment is present in the input data file

        :param l1: Input Level-1 object

        :return: a list of Level-1 objects.
        """

        # Identify connected non-ocean segments within the orbit
        ocean = l1.surface_type.get_by_name("ocean")
        not_ocean_flag = np.logical_not(ocean.flag)
        segments_len, segments_start, not_ocean = rle(not_ocean_flag)
        landseg_index = np.where(not_ocean)[0]

        # no non-ocean segments, return full segment
        if len(landseg_index) == 0:
            return [l1]

        # Test if non-ocean segments above the size threshold that will require a split of the segment.
        # The motivation behind this step to keep l1p data files as small as possible, while tolerating
        # smaller non-ocean sections
        treshold = self.cfg.polar_ocean.allow_nonocean_segment_nrecords
        large_landsegs_index = np.where(segments_len[landseg_index] > treshold)[0]
        large_landsegs_index = landseg_index[large_landsegs_index]

        # no segment split necessary, return full segment
        if len(large_landsegs_index) == 0:
            return [l1]

        # Split of orbit segment required, generate individual Level-1 segments from the ocean segments
        l1_segments = []
        start_index = 0
        for index in large_landsegs_index:
            stop_index = segments_start[index]
            subset_list = np.arange(start_index, stop_index)
            l1_segments.append(l1.extract_subset(subset_list))
            start_index = segments_start[index + 1]

        # Extract the last subset
        last_subset_list = np.arange(start_index, len(ocean.flag))
        l1_segments.append(l1.extract_subset(last_subset_list))

        # Return a list of segments
        return l1_segments

    def split_at_time_discontinuities(self, l1_list: List["Level1bData"]) -> List["Level1bData"]:
        """
        Split l1 object(s) at discontinuities of the timestamp value and return the expanded list with l1 segments.

        :param l1_list: [list] a list of l1b_files
        :return: expanded list
        """

        # Prepare input (should always be list)
        seconds_threshold = self.cfg.timestamp_discontinuities.split_at_time_gap_seconds
        dt_threshold = timedelta(seconds=seconds_threshold)

        # Output (list with l1b segments)
        l1_segments = []

        for l1 in l1_list:

            # Get timestamp discontinuities (if any)
            time = l1.time_orbit.timestamp

            # Get start/stop indices pairs
            segments_start = np.array([0])
            segments_start_indices = np.where(np.ediff1d(time) > dt_threshold)[0] + 1
            segments_start = np.append(segments_start, segments_start_indices)

            segments_stop = segments_start[1:] - 1
            segments_stop = np.append(segments_stop, len(time) - 1)

            # Check if only one segment found
            if len(segments_start) == 1:
                l1_segments.append(l1)
                continue

            # Extract subsets
            segment_indices = zip(segments_start, segments_stop)
            for start_index, stop_index in segment_indices:
                subset_indices = np.arange(start_index, stop_index + 1)
                l1_segment = l1.extract_subset(subset_indices)
                l1_segments.append(l1_segment)

        return l1_segments


class Level1PreProcessor(object):
    """
    The main Level-1 pre-processor class. The purpose of this processor is to
    pre-process radar altimeter source data for the generation of higher level
    processing levels with pysiral. Specifically the Level-1 pre-processors
    extracts polar ocean segments from the source data. It also merges data from
    different files if required to create continuous orbit segments over the
    poles.

    The Level-1 preprocessor requires a source data identifier known to pysiral
    and the processor configuration data model.

    Alternative initialization class methods:

    - `from_id()`: This class method will initialize the Level-1 preprocessor configuration
      data model from the pysiral package configuration

    Class Properties:

    - source_file_discovery:
        Class for obtaining file lists with limited metadata (time and latitude coverage)
        for a given data range. There must be one source file discovery class
        for each input data set, and this class must be a subclass of
        pysiral.l1preproc.SourceFileDiscovery

    - source_loader:
        Class for loads the content of a source data file into a Level-1 data container.

    - output_handler:
       Class for netCDF export of merged L1 data objects

    - processor_item_dict:
        Dictionary containing initialized processor items sorted by
        the intended stage in the processor pipeline

    :param source_dataset_id: source data identifier. Must be known to pysiral.
    :param cfg: Level-1 preprocessor configuration data model.
    :param hemisphere: List of hemispheres (`["nh"]`, `["sh"]`, or `['nh', 'sh']`
    :param log_directory: (Optional) Directory for the output log will be written.
    :param ctlg_directory: (Optional Directory for the output catalog
    :param source_discovery_kwargs: (Optional) dictionaary with keyword arguments
        for the source data discovery
    :param source_loader_kwargs:(Optional) dictionary with keyword arguments
        for the source loader
    """

    def __init__(
            self,
            source_dataset_id: Union[str, SourceDataID],
            cfg: L1pProcessorConfig,
            hemisphere: List[Literal["nh", "sh"]] = None,
            log_directory: Path = None,
            ctlg_directory: Path = None,
            source_discovery_kwargs: Dict = None,
            source_loader_kwargs: Dict = None
    ):
        """
        Class initialization
        """

        # Input to this class
        self.source_dataset_id = self._validate_source_dataset_id(source_dataset_id)
        self.cfg = cfg
        self.hemisphere = self._validate_hemisphere(hemisphere)
        self.log_directory = log_directory
        self.ctlg_directory = ctlg_directory

        # --- Class Properties ---

        source_discovery_kwargs = {} if source_discovery_kwargs is None else source_discovery_kwargs
        self.source_file_discovery = SourceFileDiscovery.get_cls(
            self.source_dataset_id.version_str,
            **source_discovery_kwargs
        )

        source_loader_kwargs = {} if source_loader_kwargs is None else source_loader_kwargs
        self.source_loader = SourceDataLoader.get_cls(
            self.source_dataset_id.version_str,
            **source_loader_kwargs
        )
        self.polar_ocean_segments = PolarOceanSegments(**cfg.level1_preprocessor.polar_ocean.dict())
        self.output_handler = self._get_output_handler()
        self.processor_item_dict = self._init_processor_items()

    @classmethod
    def from_ids(
            cls,
            source_dataset_id: Union[str, SourceDataID],
            l1p_id: Optional[str] = None,
            **kwargs
    ) -> "Level1PreProcessor":
        """
        Initialize the Level-1 pre-processor from source data id and an optional
        l1p id. This class initialisation methods uses the pysiral package configuration to
        select and parse Level-1 pre-processor

        :param source_dataset_id:
        :param l1p_id:

        :return: Initialized class instance
        """
        logger.info("Set Level-1 preprocessor configuration:")
        l1p_settings_filepath = psrlcfg.procdef.get_l1_from_dataset_id(source_dataset_id, l1p_id)
        cfg = L1pProcessorConfig.from_yaml(l1p_settings_filepath)
        return cls(source_dataset_id, cfg, **kwargs)

    def process_period(
            self,
            requested_period: DatePeriod
    ) -> None:
        """
        Run the Level-1 PreProcessor for a defined period.

        :param requested_period: A dateperiods.DatePeriod instance

        :return: None
        """

        # Get the list of files
        source_data_files = self.source_file_discovery.query_period(requested_period)
        logger.info(
            f"Found {len(source_data_files)} files for {self.source_dataset_id.version_str} "
            f"from {requested_period.date_label}"
        )
        self.preprocess_files(source_data_files)

    def preprocess_files(self, source_data_files: List[Path]) -> None:
        """
        Main workflow loop of the Level-1 preprocessor

        :param source_data_files: A list of files to pre-processe

        :return: None
        """

        # Validity Check
        n_input_files = len(source_data_files)
        if n_input_files == 0:
            logger.error("Passed empty input file list to preprocess_files()")
            return

        # Init helpers
        prgs = ProgressIndicator(n_input_files)

        # # A class that is passed to the input adapter to check if the pre-processor wants the
        # # content of the current file
        # polar_ocean_check = L1PreProcPolarOceanCheck(self.polar_ocean_props)

        # The stack of connected l1 segments is a list of l1 objects that together form a
        # continuous trajectory over polar oceans. This stack will be emptied and its content
        # exported to a l1p netcdf if the next segment is not connected to the stack.
        # This stack is necessary, because the polar ocean segments from the next file may
        # be connected to polar ocean segments from the previous file.
        l1_connected_stack = []

        # orbit segments may or may not be connected, therefore the list of input file
        # needs to be processed sequentially.
        for i, source_data_file in enumerate(source_data_files):

            logger.info(f"+ Process input file {prgs.get_status_report(i)} [{source_data_file.name}]")
            l1_source = self._load_source_data(source_data_file)
            l1_po_segments = self._get_source_data_polar_ocean_segments(l1_source)
            l1_export_list, l1_connected_stack = self._get_merged_segments(l1_connected_stack, l1_po_segments)

            if not l1_export_list:
                continue
            if psrlcfg.debug_mode:
                l1p_debug_map(l1_export_list, title="Polar Ocean Segments - Export")

            # Step 4: Processor items post
            # Computational expensive post-processing (e.g. computation of waveform shape parameters) can now be
            # executed as the Level-1 segments are cropped to the minimal length.
            self.l1_apply_processor_items(l1_export_list, "post_merge")

            # Step 5: Export
            for l1_export in l1_export_list:
                self.l1_export_to_netcdf(l1_export)

            if psrlcfg.debug_mode:
                l1p_debug_map(l1_connected_stack, title="Stack after Export")

        # Step : Export the last item in the stack (if it exists)
        # Stack is clean -> return
        if len(l1_connected_stack) == 0:
            pass

        # Single stack item -> export and return
        elif len(l1_connected_stack) == 1:
            self.l1_export_to_netcdf(l1_connected_stack[-1])

        # A case with more than 1 stack item is an error
        else:
            raise ValueError("something went wrong here")

    def _load_source_data(self, source_data_file: Path) -> Optional[Level1bData]:
        """
        Extract polar ocean segments from a source data file.

        :param source_data_file:
        :return:
        """

        # Step 1: Read Input
        # Map the entire orbit segment into on Level-1 data object. This is the task
        # of the input adaptor. The input handler gets only the filename and the target
        # region to assess whether it is necessary to parse and transform the file content
        # for the sake of computational efficiency.
        l1 = self.source_loader.get_l1(
            source_data_file,
            polar_ocean_check=self.polar_ocean_segments.check
        )
        if l1 is None:
            logger.info("- No polar ocean data for curent job -> skip file")
            return None

        if psrlcfg.debug_mode:
            l1p_debug_map([l1], title="Source File")

        # Step 2: Apply processor items on source data
        # for the sake of computational efficiency.
        self.l1_apply_processor_items(l1, "post_source_file")

        return l1

    def _get_source_data_polar_ocean_segments(self, l1_source: Level1bData) -> List[Level1bData]:
        """
        Extract polar ocean segments from a source data file. The input files may
        contain unwanted data (low latitude/land segments). It is the job of the
        L1PReProc children class to return only the relevant segments over polar ocean
        as a list of l1 objects.

        :param l1_source: A single L1 data object (usually directly
            loaded from a source data file)

        :return: list of L1 polar ocean segments
        """

        # Step 3: Extract and subset

        l1_po_segments = self.polar_ocean_segments.extract(l1_source)
        if psrlcfg.debug_mode:
            l1p_debug_map(l1_po_segments, title="Polar Ocean Segments")
        self.l1_apply_processor_items(l1_po_segments, "post_ocean_segment_extraction")

        return l1_po_segments

    def _get_merged_segments(
            self,
            l1_connected_stack: List[Level1bData],
            l1_polar_ocean_segments: List[Level1bData]
    ) -> Tuple[List[Level1bData], List[Level1bData]]:
        """

        :param l1_connected_stack:
        :param l1_polar_ocean_segments:

        :return:
        """
        breakpoint()

    @staticmethod
    def _validate_source_dataset_id(
            source_dataset_id: Union[str, SourceDataID]
    ) -> SourceDataID:
        """
        Ensures that source data set id is always of type SourceDataID

        :param source_dataset_id:

        :return: source dataset identifier as SourceDataID intance
        """
        return (
            source_dataset_id if isinstance(source_dataset_id, SourceDataID) else
            SourceDataID.from_str(source_dataset_id)
        )

    @staticmethod
    def _validate_hemisphere(hemisphere: Any) -> List[Literal["nh", "sh"]]:
        """
        Ensure that value of hemisphere value is list of "nh" and "sh"

        :param hemisphere: class input for hemisphere keyword

        :raises ValueError: Invalid hemisphere definition

        :return: validated list of target hemispheres
        """

        # Default is global
        if hemisphere is None:
            return ["nh", "sh"]

        if isinstance(hemisphere, str):
            hemisphere = [hemisphere]

        error_msg = f"Invalid {hemisphere=} (Must be list with literals `nh` and `sh`"
        if not isinstance(hemisphere, list):
            raise ValueError(error_msg)

        if any(h not in ["nh", "sh"] for h in hemisphere):
            raise ValueError(error_msg)

        return hemisphere

    def _init_processor_items(self) -> Dict[str, list]:
        """
        Populate the processor item dictionary with initialized processor item classes
        for the different stages of the Level-1 pre-processor.

        This method will evaluate the configuration dictionary passed to this class,
        retrieve the corresponding classes, initialize them and store in a dictionary.

        Dictionary entries are the names of the processing stages and the content
        is a list of classes (cls, label) per processing stage.

        :return: A dictionary with a list of Level-1 PreProcessor items depending
            on processor stage
        """
        processor_item_dict = defaultdict(list)
        for processing_item_definition in self.cfg.level1_preprocessor.processing_items:
            def_ = L1PProcItemDef.from_l1procdef_dict(processing_item_definition.dict())
            cls = def_.get_initialized_processing_item_instance()
            processor_item_dict[def_.stage].append((cls, def_.label,))
        return processor_item_dict

    def _get_output_handler(self) -> Level1POutputHandler:
        """
        Return the initialized output handler

        :return:
        """
        file_version = DataVersion(self.cfg.pysiral_package_config.version).filename
        return Level1POutputHandler(
            self.source_dataset_id.platform_or_mission,
            self.source_dataset_id.timeliness,
            self.cfg.pysiral_package_config.id,
            file_version
        )

    def l1_export_to_netcdf(self, l1: "Level1bData") -> None:
        """
        Exports the Level-1 object as l1p netCDF

        :param l1: The Level-1 object to exported

        :return:
        """
        minimum_n_records = self.cfg.level1_preprocessor.export_minimum_n_records
        if l1.n_records >= minimum_n_records:
            self.output_handler.export_to_netcdf(l1)
            logger.info(f"- Written l1p product: {self.output_handler.last_written_file}")
        else:
            logger.warning(f"- Orbit segment below {minimum_n_records=} ({l1.n_records}), skipping")

    def l1_apply_processor_items(
            self,
            l1: Union["Level1bData", List["Level1bData"]],
            stage_name: str
    ) -> None:
        """
        Apply the processor items defined in the l1 processor configuration file
        to either a l1 data object or a list of l2 data objects at a defined
        stage of the processor.

        This method is a wrapper that deals with multiple input types.
        The functionality is implemented in `_l1_apply_proc_item`

        :param l1: Level-1 data object or list of Level-1 data objects
        :param stage_name: Name of the processing stage. Valid options are
            (`post_source_file`, `post_ocean_segment_extraction`, `post_merge`)

        :return: None, the l1_segments are changed in place
        """

        # Check if there is anything to do first
        logger.info(f"- Apply {stage_name} processing items")
        if stage_name not in self.processor_item_dict:
            return

        (
            [self._l1_apply_proc_item(l1_item, stage_name) for l1_item in l1] if isinstance(l1, list)
            else self._l1_apply_proc_item(l1, stage_name)
        )

    @debug_timer("L1 processor items")
    def _l1_apply_proc_item_list(self,
                                 l1_list: List["Level1bData"],
                                 stage_name: str
                                 ) -> None:
        """
        Apply processor item to full stack

        :param l1_list:
        :param stage_name:

        :return:
        """
        for procitem, label in self.processor_item_dict.get(stage_name, []):
            procitem.apply_list(l1_list)

    @debug_timer("L1 processor items")
    def _l1_apply_proc_item(self, l1: "Level1bData", stage_name: str) -> None:
        """
        Apply l1 processor items

        :param l1: The L1 data containier
        :param stage_name: processor stage
        """
        for procitem, label in self.processor_item_dict.get(stage_name, []):
            procitem.apply(l1)

# class L1PreProcBase(object):
#
#     def __init__(
#             self,
#             cls_name: str,
#             input_adapter: L1PInputCLS,
#             output_handler: Level1POutputHandler,
#             cfg: Dictionary
#     ) -> None:
#
#         # Make sure the logger/error handler has the name of the parent class
#         super(L1PreProcBase, self).__init__(cls_name)
#
#         # The class that translates a given input file into an L1BData object
#         self.input_adapter = input_adapter
#
#         # Output data handler that creates l1p netCDF files from l1 data objects
#         self.output_handler = output_handler
#
#         # The configuration for the pre-processor
#         self.cfg = cfg
#
#         # Initialize the L1 processor items
#         self.processor_item_dict = {}
#         self._init_processor_items()
#
#         # # The stack of Level-1 objects is a simple list
#         # self.l1_stack = []
#
#     def _init_processor_items(self) -> None:
#         """
#         Popuplate the processor item dictionary with initialized processor item classes
#         for the different stages of the Level-1 pre-processor.
#
#         This method will evaluate the configuration dictionary passed to this class,
#         retrieve the corresponding classes, initialize them and store in a dictionary.
#
#         Dictionary entries are the names of the processing stages and the content
#         is a list of classes (cls, label) per processing stage.
#
#         :return:
#         """
#
#         for processing_item_definition_dict in self.cfg.processing_items:
#             def_ = L1PProcItemDef.from_l1procdef_dict(processing_item_definition_dict)
#             cls = def_.get_initialized_processing_item_instance()
#             if def_.stage in self.processor_item_dict:
#                 self.processor_item_dict[def_.stage].append((cls, def_.label,))
#             else:
#                 self.processor_item_dict[def_.stage] = [(cls, def_.label)]
#
#     def process_input_files(self, input_file_list: List[Union[Path, str]]) -> None:
#         """
#         Main entry point for the Level-1 pre-processor and start of the main
#         loop of this class:
#
#         1. Sequentially reads source file from input list of source files
#         2. Applies processing items (stage: `post_source_file`)
#         3. Extract polar ocean segments (can be multiple per source file)
#         4. Applies processing items (stage: `post_polar_ocean_segment_extraction`)
#         5. Creates a stack of spatially connecting segments
#            a. if this includes all segments of the source file the loop
#               progresses to the next source file
#            b. if the stack contains a spatially unconnected segments
#                 1. merge all connected segments to a single l1 object
#                 2. Applies processing items (stage: `post_merge`)
#                 3. Export to l1p netCDF
#         6. Export the last l1p segment at the end of the loop.
#
#         :param input_file_list: A list full filepath for the pre-processor
#
#         :return: None
#         """
#
#         # Validity Check
#         n_input_files = len(input_file_list)
#         if n_input_files == 0:
#             logger.warning("Passed empty input file list to process_input_files()")
#             return
#
#         # Init helpers
#         prgs = ProgressIndicator(n_input_files)
#
#         # A class that is passed to the input adapter to check if the pre-processor wants the
#         # content of the current file
#         polar_ocean_check = L1PreProcPolarOceanCheck(self.polar_ocean_props)
#
#         # The stack of connected l1 segments is a list of l1 objects that together form a
#         # continuous trajectory over polar oceans. This stack will be emptied and its content
#         # exported to a l1p netcdf if the next segment is not connected to the stack.
#         # This stack is necessary, because the polar ocean segments from the next file may
#         # be connected to polar ocean segments from the previous file.
#         l1_connected_stack = []
#
#         # orbit segments may or may not be connected, therefore the list of input file
#         # needs to be processed sequentially.
#         for i, input_file in enumerate(input_file_list):
#
#             # Step 1: Read Input
#             # Map the entire orbit segment into on Level-1 data object. This is the task
#             # of the input adaptor. The input handler gets only the filename and the target
#             # region to assess whether it is necessary to parse and transform the file content
#             # for the sake of computational efficiency.
#             logger.info(f"+ Process input file {prgs.get_status_report(i)} [{input_file.name}]")
#             l1 = self.input_adapter.get_l1(input_file, polar_ocean_check=polar_ocean_check)
#             if l1 is None:
#                 logger.info("- No polar ocean data for curent job -> skip file")
#                 continue
#             if __debug__:
#                 l1p_debug_map([l1], title="Source File")
#
#             # Step 2: Apply processor items on source data
#             # for the sake of computational efficiency.
#             self.l1_apply_processor_items(l1, "post_source_file")
#
#             # Step 3: Extract and subset
#             # The input files may contain unwanted data (low latitude/land segments).
#             # It is the job of the L1PReProc children class to return only the relevant
#             # segments over polar ocean as a list of l1 objects.
#             l1_po_segments = self.extract_polar_ocean_segments(l1)
#             if __debug__:
#                 l1p_debug_map(l1_po_segments, title="Polar Ocean Segments")
#
#             # Optional Step 4 (needs to be specifically activated in l1 processor config file)
#             # NOTE: This is here because there are files in the Sentinel-3AB thematic sea ice product
#             #       that contain both lrm and sar data.
#             detect_radar_mode_change = self.cfg.get("detect_radar_mode_change", False)
#             if detect_radar_mode_change:
#                 l1_po_segments = self.l1_split_radar_mode_segments(l1_po_segments)
#
#             self.l1_apply_processor_items(l1_po_segments, "post_ocean_segment_extraction")
#
#             # Step 5: Merge orbit segments
#             # Add the list of orbit segments to the l1 data stack and merge those that
#             # are connected (e.g. two half orbits connected at the pole) into a single l1
#             # object. Orbit segments that  are unconnected from other segments in the stack
#             # will be exported to netCDF files.
#             l1_export_list, l1_connected_stack = self.l1_get_output_segments(l1_connected_stack, l1_po_segments)
#
#             # l1p_debug_map(l1_connected_stack, title="Polar Ocean Segments - Stack")
#             if not l1_export_list:
#                 continue
#             if __debug__:
#                 l1p_debug_map(l1_export_list, title="Polar Ocean Segments - Export")
#
#             # Step 4: Processor items post
#             # Computational expensive post-processing (e.g. computation of waveform shape parameters) can now be
#             # executed as the Level-1 segments are cropped to the minimal length.
#             self.l1_apply_processor_items(l1_export_list, "post_merge")
#
#             # Step 5: Export
#             for l1_export in l1_export_list:
#                 self.l1_export_to_netcdf(l1_export)
#
#             if __debug__:
#                 l1p_debug_map(l1_connected_stack, title="Stack after Export")
#
#         # Step : Export the last item in the stack (if it exists)
#         # Stack is clean -> return
#         if len(l1_connected_stack) == 0:
#             return
#
#         # Single stack item -> export and return
#         elif len(l1_connected_stack) == 1:
#             self.l1_export_to_netcdf(l1_connected_stack[-1])
#             return
#
#         # A case with more than 1 stack item is an error
#         else:
#             raise ValueError("something went wrong here")
#
#     def extract_polar_ocean_segments(self, l1: "Level1bData") -> List["Level1bData"]:
#         """
#         Needs to be implemented by child classes, because the exact algorithm
#         depends on th source data.
#
#         :param l1:
#
#         :return: None
#
#         :raises: NotImplementedError
#         """
#         raise NotImplementedError("")
#
#     def l1_apply_processor_items(self,
#                                  l1: Union["Level1bData", List["Level1bData"]],
#                                  stage_name: str
#                                  ) -> None:
#         """
#         Apply the processor items defined in the l1 processor configuration file
#         to either a l1 data object or a list of l2 data objects at a defined
#         stage of the procesor.
#
#         This method is a wrapper that deals with multiple input types.
#         The functionality is implemented in `_l1_apply_proc_item`
#
#         :param l1: Level-1 data object or list of Level-1 data objects
#         :param stage_name: Name of the processing stage. Valid options are
#             (`post_source_file`, `post_ocean_segment_extraction`, `post_merge`)
#
#         :return: None, the l1_segments are changed in place
#         """
#
#         # Check if there is anything to do first
#         logger.info(f"- Apply {stage_name} processing items")
#         if stage_name not in self.processor_item_dict:
#             return
#
#         (
#             [self._l1_apply_proc_item(l1_item, stage_name) for l1_item in l1] if isinstance(l1, list)
#             else self._l1_apply_proc_item(l1, stage_name)
#         )
#
#     @debug_timer("L1 processor items")
#     def _l1_apply_proc_item_list(self,
#                                  l1_list: List["Level1bData"],
#                                  stage_name: str
#                                  ) -> None:
#         """
#         Apply processor item to full stack
#
#         :param l1_list:
#         :param stage_name:
#
#         :return:
#         """
#         for procitem, label in self.processor_item_dict.get(stage_name, []):
#             procitem.apply_list(l1_list)
#
#     @debug_timer("L1 processor items")
#     def _l1_apply_proc_item(self, l1: "Level1bData", stage_name: str) -> None:
#         """
#         Apply l1 processor items
#
#         :param l1: The L1 data containier
#         :param stage_name: processor stage
#         """
#         for procitem, label in self.processor_item_dict.get(stage_name, []):
#             procitem.apply(l1)
#
#     def l1_get_output_segments(self,
#                                l1_connected_stack: List["Level1bData"],
#                                l1_po_segments: List["Level1bData"]
#                                ) -> Tuple[List["Level1bData"], List["Level1bData"]]:
#         """
#         This method sorts the stack of connected l1 segments and the polar ocean segments
#         of the most recent file and sorts the segments into (connected) output and
#         the new stack, defined by the last (connected) item.
#
#         :param l1_connected_stack: List of connected L1 polar ocean segments (from previous file(s))
#         :param l1_po_segments: List of L1 polar ocean segments (from current file)
#
#         :return: L1 segments for the ouput, New l1 stack of connected items
#         """
#
#         # Create a list of all currently available l1 segments. This includes the
#         # elements from the stack and the polar ocean segments
#         all_l1_po_segments = [*l1_connected_stack, *l1_po_segments]
#
#         # There is a number of case to be caught her.
#         # Case 1: The list of polar ocean segments might be empty
#         if not l1_po_segments:
#             return l1_connected_stack, l1_po_segments
#
#         # Case 2: If there is only one element, all data goes to the stack
#         # and no data needs to be exported.
#         # (because the next file could be connected to it)
#         if len(all_l1_po_segments) == 1:
#             return [], l1_po_segments
#
#         # Check if segment is connected to the next one
#         are_connected = [
#             self.l1_are_connected(l1_0, l1_1)
#             for l1_0, l1_1 in zip(all_l1_po_segments[:-1], all_l1_po_segments[1:])
#         ]
#
#         # Create a list of (piece-wise) merged elements.
#         # The last element of this list will be the transferred to the
#         # next iteration of the file list
#         merged_l1_list = [copy.deepcopy(all_l1_po_segments[0])]
#         for idx, is_connected in enumerate(are_connected):
#             target_l1 = all_l1_po_segments[idx + 1]
#             if is_connected:
#                 merged_l1_list[-1].append(target_l1, remove_overlap=True)
#             else:
#                 merged_l1_list.append(copy.deepcopy(target_l1))
#
#         # Return (elements marked for export, l1_connected stack)
#         return merged_l1_list[:-1], [merged_l1_list[-1]]
#
#     @staticmethod
#     def l1_split_radar_mode_segments(l1_list: List["Level1bData"]) -> List["Level1bData"]:
#         """
#         This method checks if there is more than one radar mode per l1 segment and
#         splits the segments for each radar mode change. This is done, because
#         multiple radar modes per segment may be problematic for Level-1 pre-processor
#         items. The final l1p output file may contain multiple radar modes.
#
#         NOTE: So far this is only needed for the Sentinel-3A/B L2 thematic sea ice product.
#               No other files have been found so far that have more than one radar mode.
#
#         :param l1_list: Input list of l1 segments.
#
#         :return: l1_output_list: Output list of l1 segments with single radar mode
#         """
#         output_l1_list = []
#         for l1 in l1_list:
#
#             if l1.waveform.num_radar_modes > 1:
#
#                 # Get a list of start and end index
#                 change_idxs = np.where(np.ediff1d(l1.waveform.radar_mode))[0] + 1
#
#                 segment_start_idxs = np.insert(change_idxs, 0, 0)
#                 segment_end_idxs = np.insert(change_idxs - 1, len(change_idxs), l1.n_records - 1)
#
#                 for start_idx, end_idx in zip(segment_start_idxs, segment_end_idxs):
#                     l1_radar_mode_subset = l1.extract_subset(np.arange(start_idx, end_idx))
#                     output_l1_list.append(l1_radar_mode_subset)
#
#             else:
#                 output_l1_list.append(l1)
#
#         return output_l1_list
#
#     def l1_are_connected(self, l1_0: "Level1bData", l1_1: "Level1bData") -> bool:
#         """
#         Check if the start time of l1 segment 1 and the stop time of l1 segment 0
#         indicate neighbouring orbit segments.
#         -> Assumes explicitly that l1_0 comes before l1_1
#
#         :param l1_0:
#         :param l1_1:
#
#         :return: Flag if l1 segments are connected (True of False)
#         """
#
#         # test alternate way of checking connectivity (distance)
#         l1_0_last_latlon = l1_0.time_orbit.latitude[-1], l1_0.time_orbit.longitude[-1]
#         l1_1_first_latlon = l1_1.time_orbit.latitude[0], l1_1.time_orbit.longitude[0]
#         distance_km = distance.distance(l1_0_last_latlon, l1_1_first_latlon).km
#         logger.debug(f"- distance_km={distance_km}")
#
#         # Test if segments are adjacent based on time gap between them
#         tdelta = l1_1.info.start_time - l1_0.info.stop_time
#         threshold = self.cfg.orbit_segment_connectivity.max_connected_segment_timedelta_seconds
#         return tdelta.total_seconds() <= threshold
#
#     def l1_export_to_netcdf(self, l1: "Level1bData") -> None:
#         """
#         Exports the Level-1 object as l1p netCDF
#
#         :param l1: The Level-1 object to exported
#
#         :return:
#         """
#         minimum_n_records = self.cfg.get("export_minimum_n_records", 0)
#         if l1.n_records >= minimum_n_records:
#             self.output_handler.export_to_netcdf(l1)
#             logger.info(f"- Written l1p product: {self.output_handler.last_written_file}")
#         else:
#             logger.warning("- Orbit segment below minimum size (%g), skipping" % l1.n_records)
#
#     def trim_single_hemisphere_segment_to_polar_region(self, l1: "Level1bData") -> "Level1bData":
#         """
#         Extract polar region of interest from a segment that is either north or south (not global)
#
#         :param l1: Input Level-1 object
#
#         :return: Trimmed Input Level-1 object
#         """
#         polar_threshold = self.cfg.polar_ocean.polar_latitude_threshold
#         is_polar = np.abs(l1.time_orbit.latitude) >= polar_threshold
#         polar_subset = np.where(is_polar)[0]
#         if len(polar_subset) != l1.n_records:
#             l1.trim_to_subset(polar_subset)
#         return l1
#
#     def trim_two_hemisphere_segment_to_polar_regions(self, l1: "Level1bData"
#                                                      ) -> Union[None, List["Level1bData"]]:
#         """
#         Extract polar regions of interest from a segment that is either north, south or both. The method will
#         preserve the order of the hemispheres
#
#         :param l1: Input Level-1 object
#         :return: List of Trimmed Input Level-1 objects
#         """
#
#         polar_threshold = self.cfg.polar_ocean.polar_latitude_threshold
#         l1_list = []
#
#         # Loop over the two hemispheres
#         for hemisphere in self.cfg.polar_ocean.target_hemisphere:
#
#             if hemisphere == "north":
#                 is_polar = l1.time_orbit.latitude >= polar_threshold
#
#             elif hemisphere == "south":
#                 is_polar = l1.time_orbit.latitude <= (-1.0 * polar_threshold)
#
#             else:
#                 raise ValueError(f"Unknown {hemisphere=} [north|south]")
#
#             # Extract the subset (if applicable)
#             polar_subset = np.where(is_polar)[0]
#             n_records_subset = len(polar_subset)
#
#             # is true subset -> add subset to output list
#             if n_records_subset != l1.n_records and n_records_subset > 0:
#                 l1_segment = l1.extract_subset(polar_subset)
#                 l1_list.append(l1_segment)
#
#             # entire segment in polar region -> add full segment to output list
#             elif n_records_subset == l1.n_records:
#                 l1_list.append(l1)
#
#         # Last step: Sort the list to maintain temporal order
#         # (only if more than 1 segment)
#         if len(l1_list) > 1:
#             l1_list = sorted(l1_list, key=attrgetter("tcs"))
#
#         return l1_list
#
#     def trim_multiple_hemisphere_segment_to_polar_regions(self, l1: "Level1bData"
#                                                           ) -> Union[None, List["Level1bData"]]:
#         """
#         Extract polar regions segments from an orbit segment that may cross from north to south
#         to north again (or vice versa).
#
#         :param l1: Input Level-1 object
#
#         :return: List of Trimmed Input Level-1 objects
#         """
#
#         # Compute flag for segments in polar regions
#         # regardless of hemisphere
#         polar_threshold = self.cfg.polar_ocean.polar_latitude_threshold
#         is_polar = np.array(np.abs(l1.time_orbit.latitude) >= polar_threshold)
#
#         # Find start and end indices of continuous polar
#         # segments based on the change of the `is_polar` flag
#         change_to_polar = np.ediff1d(is_polar.astype(int))
#         change_to_polar = np.insert(change_to_polar, 0, 1 if is_polar[0] else 0)
#         change_to_polar[-1] = -1 if is_polar[-1] else change_to_polar[-1]
#         start_idx = np.where(change_to_polar > 0)[0]
#         end_idx = np.where(change_to_polar < 0)[0]
#
#         # Create a list of l1 subsets
#         l1_list = []
#         for i in np.arange(len(start_idx)):
#             polar_idxs = np.arange(start_idx[i], end_idx[i])
#             l1_segment = l1.extract_subset(polar_idxs)
#             l1_list.append(l1_segment)
#
#         return l1_list
#
#     def trim_full_orbit_segment_to_polar_regions(self, l1: "Level1bData") -> Union[None, List["Level1bData"]]:
#         """
#         Extract polar regions of interest from a segment that is either north, south or both. The method will
#         preserve the order of the hemispheres
#
#         :param l1: Input Level-1 object
#         :return: List of Trimmed Input Level-1 objects
#         """
#
#         polar_threshold = self.cfg.polar_ocean.polar_latitude_threshold
#         l1_list = []
#
#         # Loop over the two hemispheres
#         for hemisphere in self.cfg.polar_ocean.target_hemisphere:
#
#             # Compute full polar subset range
#             if hemisphere == "north":
#                 is_polar = l1.time_orbit.latitude >= polar_threshold
#             elif hemisphere == "south":
#                 is_polar = l1.time_orbit.latitude <= (-1.0 * polar_threshold)
#             else:
#                 raise ValueError(f"Unknown {hemisphere=} [north|south]")
#
#             # Step: Extract the polar ocean segment for the given hemisphere
#             polar_subset = np.where(is_polar)[0]
#             n_records_subset = len(polar_subset)
#
#             # Safety check
#             if n_records_subset == 0:
#                 continue
#             l1_segment = l1.extract_subset(polar_subset)
#
#             # Step: Trim non-ocean segments
#             l1_segment = self.trim_non_ocean_data(l1_segment)
#
#             # Step: Split the polar subset to its marine regions
#             l1_segment_list = self.split_at_large_non_ocean_segments(l1_segment)
#
#             # Step: append the ocean segments
#             l1_list.extend(l1_segment_list)
#
#         # Last step: Sort the list to maintain temporal order
#         # (only if more than 1 segment)
#         if len(l1_list) > 1:
#             l1_list = sorted(l1_list, key=attrgetter("tcs"))
#
#         return l1_list
#
#     def filter_small_ocean_segments(self, l1: "Level1bData") -> "Level1bData":
#         """
#         This method sets the surface type flag of very small ocean segments to land.
#         This action should prevent large portions of land staying in the l1 segment
#         is a small fjord et cetera is crossed. It should also filter out smaller
#         ocean segments that do not have a realistic chance of freeboard retrieval.
#
#         :param l1: A pysiral.l1bdata.Level1bData instance
#
#         :return: filtered l1 object
#         """
#
#         # Minimum size for valid ocean segments
#         ocean_mininum_size_nrecords = self.cfg.polar_ocean.ocean_mininum_size_nrecords
#
#         # Get the clusters of ocean parts in the l1 object
#         ocean_flag = l1.surface_type.get_by_name("ocean").flag
#         land_flag = l1.surface_type.get_by_name("land").flag
#         segments_len, segments_start, not_ocean = rle(ocean_flag)
#
#         # Find smaller than threshold ocean segments
#         small_cluster_indices = np.where(segments_len < ocean_mininum_size_nrecords)[0]
#
#         # Do not mess with the l1 object if not necessary
#         if len(small_cluster_indices == 0):
#             return l1
#
#         # Set land flag -> True for small ocean segments
#         for small_cluster_index in small_cluster_indices:
#             i0 = segments_start[small_cluster_index]
#             i1 = i0 + segments_len[small_cluster_index]
#             land_flag[i0:i1] = True
#
#         # Update the l1 surface type flag by re-setting the land flag
#         l1.surface_type.add_flag(land_flag, "land")
#
#         # All done
#         return l1
#
#     @staticmethod
#     def trim_non_ocean_data(l1: "Level1bData") -> Union[None, "Level1bData"]:
#         """
#         Remove leading and trailing data that is not if type ocean.
#
#         :param l1: The input Level-1 objects
#
#         :return: The subsetted Level-1 objects. (Segments with no ocean data are removed from the list)
#         """
#
#         ocean = l1.surface_type.get_by_name("ocean")
#         first_ocean_index = get_first_array_index(ocean.flag, True)
#         last_ocean_index = get_last_array_index(ocean.flag, True)
#         if first_ocean_index is None or last_ocean_index is None:
#             return None
#         n = l1.info.n_records - 1
#         is_full_ocean = first_ocean_index == 0 and last_ocean_index == n
#         if not is_full_ocean:
#             ocean_subset = np.arange(first_ocean_index, last_ocean_index + 1)
#             l1.trim_to_subset(ocean_subset)
#         return l1
#
#     def split_at_large_non_ocean_segments(self, l1: "Level1bData") -> List["Level1bData"]:
#         """
#         Identify larger segments that are not ocean (land, land ice) and split the segments if necessary.
#         The return value will always be a list of Level-1 object instances, even if no non-ocean data
#         segment is present in the input data file
#
#         :param l1: Input Level-1 object
#
#         :return: a list of Level-1 objects.
#         """
#
#         # Identify connected non-ocean segments within the orbit
#         ocean = l1.surface_type.get_by_name("ocean")
#         not_ocean_flag = np.logical_not(ocean.flag)
#         segments_len, segments_start, not_ocean = rle(not_ocean_flag)
#         landseg_index = np.where(not_ocean)[0]
#
#         # no non-ocean segments, return full segment
#         if len(landseg_index) == 0:
#             return [l1]
#
#         # Test if non-ocean segments above the size threshold that will require a split of the segment.
#         # The motivation behind this step to keep l1p data files as small as possible, while tolerating
#         # smaller non-ocean sections
#         treshold = self.cfg.polar_ocean.allow_nonocean_segment_nrecords
#         large_landsegs_index = np.where(segments_len[landseg_index] > treshold)[0]
#         large_landsegs_index = landseg_index[large_landsegs_index]
#
#         # no segment split necessary, return full segment
#         if len(large_landsegs_index) == 0:
#             return [l1]
#
#         # Split of orbit segment required, generate individual Level-1 segments from the ocean segments
#         l1_segments = []
#         start_index = 0
#         for index in large_landsegs_index:
#             stop_index = segments_start[index]
#             subset_list = np.arange(start_index, stop_index)
#             l1_segments.append(l1.extract_subset(subset_list))
#             start_index = segments_start[index + 1]
#
#         # Extract the last subset
#         last_subset_list = np.arange(start_index, len(ocean.flag))
#         l1_segments.append(l1.extract_subset(last_subset_list))
#
#         # Return a list of segments
#         return l1_segments
#
#     def split_at_time_discontinuities(self, l1_list: List["Level1bData"]) -> List["Level1bData"]:
#         """
#         Split l1 object(s) at discontinuities of the timestamp value and return the expanded list with l1 segments.
#
#         :param l1_list: [list] a list of l1b_files
#         :return: expanded list
#         """
#
#         # Prepare input (should always be list)
#         seconds_threshold = self.cfg.timestamp_discontinuities.split_at_time_gap_seconds
#         dt_threshold = timedelta(seconds=seconds_threshold)
#
#         # Output (list with l1b segments)
#         l1_segments = []
#
#         for l1 in l1_list:
#
#             # Get timestamp discontinuities (if any)
#             time = l1.time_orbit.timestamp
#
#             # Get start/stop indices pairs
#             segments_start = np.array([0])
#             segments_start_indices = np.where(np.ediff1d(time) > dt_threshold)[0] + 1
#             segments_start = np.append(segments_start, segments_start_indices)
#
#             segments_stop = segments_start[1:] - 1
#             segments_stop = np.append(segments_stop, len(time) - 1)
#
#             # Check if only one segment found
#             if len(segments_start) == 1:
#                 l1_segments.append(l1)
#                 continue
#
#             # Extract subsets
#             segment_indices = zip(segments_start, segments_stop)
#             for start_index, stop_index in segment_indices:
#                 subset_indices = np.arange(start_index, stop_index + 1)
#                 l1_segment = l1.extract_subset(subset_indices)
#                 l1_segments.append(l1_segment)
#
#         return l1_segments
#
#     @property
#     def polar_ocean_props(self) -> Union[Dict, AttrDict]:
#         if "polar_ocean" not in self.cfg:
#             raise KeyError("Missing configuration key `polar_ocean` in Level-1 Pre-Processor Options")
#         return self.cfg.polar_ocean
#
#     @property
#     def orbit_segment_connectivity_props(self) -> Union[Dict, AttrDict]:
#         if "orbit_segment_connectivity" not in self.cfg:
#             raise KeyError("Missing configuration key `orbit_segment_connectivity` in Level-1 Pre-Processor Options")
#         return self.cfg.orbit_segment_connectivity
#
#
# class L1PreProcCustomOrbitSegment(L1PreProcBase):
#     """ A Pre-Processor for input files with arbitrary segment lenght (e.g. CryoSat-2) """
#
#     def __init__(self, *args):
#         super(L1PreProcCustomOrbitSegment, self).__init__(self.__class__.__name__, *args)
#
#     def extract_polar_ocean_segments(self, l1: "Level1bData") -> List["Level1bData"]:
#         """
#         Splits the input Level-1 object into the polar ocean segments (e.g. by trimming land at the edges
#         or by splitting into several parts if there are land masses with the orbit segment). The returned
#         polar ocean segments should be generally free of data over non-ocean parts of the orbit, except
#         for smaller parts within the orbit.
#
#         NOTE: This subclass of the Level-1 Pre-Processor is designed for input data type with arbitrary
#               orbit segment length (e.g. data of CryoSat-2 where the orbit segments of the input data
#               is controlled by the mode mask changes).
#
#         :param l1: A Level-1 data object
#
#         :return: A list of Level-1 data objects (subsets of polar ocean segments from input l1)
#         """
#
#         # Step: Filter small ocean segments
#         # NOTE: The objective is to remove any small marine regions (e.g. in fjords) that do not have any
#         #       reasonable chance of freeboard/ssh retrieval early on in the pre-processing.
#         if "ocean_mininum_size_nrecords" in self.cfg.polar_ocean:
#             logger.info("- filter ocean segments")
#             l1 = self.filter_small_ocean_segments(l1)
#
#         # Step: Trim the orbit segment to latitude range for the specific hemisphere
#         # NOTE: There is currently no case of an input data set that is not of type half-orbit and that
#         #       would have coverage in polar regions of both hemisphere. Therefore, `l1_subset` is assumed to
#         #       be a single Level-1 object instance and not a list of instances.  This needs to be changed if
#         #      `input_file_is_single_hemisphere=False`
#         logger.info("- extracting polar region subset(s)")
#         if self.cfg.polar_ocean.input_file_is_single_hemisphere:
#             l1_list = [self.trim_single_hemisphere_segment_to_polar_region(l1)]
#         else:
#             l1_list = self.trim_two_hemisphere_segment_to_polar_regions(l1)
#         logger.info(f"- extracted {len(l1_list)} polar region subset(s)")
#
#         # Step: Split the l1 segments at time discontinuities.
#         # NOTE: This step is optional. It requires the presence of the options branch `timestamp_discontinuities`
#         #       in the L1 pre-processor config file
#         if "timestamp_discontinuities" in self.cfg:
#             logger.info("- split at time discontinuities")
#             l1_list = self.split_at_time_discontinuities(l1_list)
#
#         # Step: Trim the non-ocean parts of the subset (e.g. land, land-ice, ...)
#         # NOTE: Generally it can be assumed that the l1 object passed to this method contains polar ocean data.
#         #       But there tests before only include if there is ocean data and data above the polar latitude
#         #       threshold. It can therefore happen that trimming the non-ocean data leaves an empty Level-1 object.
#         #       In this case an empty list is returned.
#         logger.info("- trim outer non-ocean regions")
#         l1_trimmed_list = []
#         for l1 in l1_list:
#             l1_trimmed = self.trim_non_ocean_data(l1)
#             if l1_trimmed is not None:
#                 l1_trimmed_list.append(l1_trimmed)
#
#         # Step: Split the remaining subset at non-ocean parts.
#         # NOTE: There is no need to split the orbit at small features. See option `allow_nonocean_segment_nrecords`
#         #       in the l1p processor definition. But even if there are no segments to split, the output will always
#         #       be a list per requirements of the Level-1 pre-processor workflow.
#         l1_list = []
#         for l1 in l1_trimmed_list:
#             l1_splitted_list = self.split_at_large_non_ocean_segments(l1)
#             l1_list.extend(l1_splitted_list)
#
#         # All done, return the list of polar ocean segments
#         return l1_list
#
#
# class L1PreProcHalfOrbit(L1PreProcBase):
#     """ A Pre-Processor for input files with a full orbit around the earth (e.g. ERS-1/2) """
#
#     def __init__(self, *args):
#         super(L1PreProcHalfOrbit, self).__init__(self.__class__.__name__, *args)
#
#     def extract_polar_ocean_segments_half(self, l1: "Level1bData") -> List["Level1bData"]:
#         """
#         Splits the input Level-1 object into the polar ocean segments (e.g. by trimming land at the edges
#         or by splitting into several parts if there are land masses with the orbit segment). The returned
#         polar ocean segments should be generally free of data over non-ocean parts of the orbit, except
#         for smaller parts within the orbit.
#
#         NOTE: This subclass of the Level-1 Pre-Processor is designed for input data type with coverage
#               from pole to pole (e.g. Envisat SGDR)
#
#         :param l1: A Level-1 data object
#
#         :return: A list of Level-1 data objects (subsets of polar ocean segments from input l1)
#         """
#
#         # Step: Filter small ocean segments
#         # NOTE: The objective is to remove any small marine regions (e.g. in fjords) that do not have any
#         #       reasonable chance of freeboard/ssh retrieval early on in the pre-processing.
#         if "ocean_mininum_size_nrecords" in self.cfg.polar_ocean:
#             logger.info("- filter ocean segments")
#             l1 = self.filter_small_ocean_segments(l1)
#
#         # Step: Extract Polar ocean segments from full orbit respecting the selected target hemisphere
#         l1_list = self.trim_two_hemisphere_segment_to_polar_regions(l1)
#         logger.info(f"- extracted {len(l1_list)} polar region subset(s)")
#
#         # Step: Split the l1 segments at time discontinuities.
#         # NOTE: This step is optional. It requires the presence of the options branch `timestamp_discontinuities`
#         #       in the L1 pre-processor config file
#         if "timestamp_discontinuities" in self.cfg:
#             logger.info("- split at time discontinuities")
#             l1_list = self.split_at_time_discontinuities(l1_list)
#
#         # Step: Trim the non-ocean parts of the subset (e.g. land, land-ice, ...)
#         # NOTE: Generally it can be assumed that the l1 object passed to this method contains polar ocean data.
#         #       But there tests before only include if there is ocean data and data above the polar latitude
#         #       threshold. It can therefore happen that trimming the non-ocean data leaves an empty Level-1 object.
#         #       In this case an empty list is returned.
#         logger.info("- trim outer non-ocean regions")
#         l1_trimmed_list = []
#         for l1 in l1_list:
#             l1_trimmed = self.trim_non_ocean_data(l1)
#             if l1_trimmed is not None:
#                 l1_trimmed_list.append(l1_trimmed)
#
#         # Step: Split the remaining subset at non-ocean parts.
#         # NOTE: There is no need to split the orbit at small features. See option `allow_nonocean_segment_nrecords`
#         #       in the l1p processor definition. But even if there are no segments to split, the output will always
#         #       be a list per requirements of the Level-1 pre-processor workflow.
#         l1_list = []
#         for l1 in l1_trimmed_list:
#             l1_splitted_list = self.split_at_large_non_ocean_segments(l1)
#             l1_list.extend(l1_splitted_list)
#
#         # All done, return the list of polar ocean segments
#         return l1_list
#
#
# class L1PreProcFullOrbit(L1PreProcBase):
#     """ A Pre-Processor for input files with a full orbit around the earth (e.g. ERS-1/2) """
#
#     def __init__(self, *args):
#         super(L1PreProcFullOrbit, self).__init__(self.__class__.__name__, *args)
#
#     def extract_polar_ocean_segments(self, l1: "Level1bData") -> List["Level1bData"]:
#         """
#         Splits the input Level-1 object into the polar ocean segments (e.g. by trimming land at the edges
#         or by splitting into several parts if there are land masses with the orbit segment). The returned
#         polar ocean segments should be generally free of data over non-ocean parts of the orbit, except
#         for smaller parts within the orbit.
#
#         NOTE: This subclass of the Level-1 Pre-Processor is designed for input data type with arbitrary
#               orbit segment length (e.g. data of CryoSat-2 where the orbit segments of the input data
#               is controlled by the mode mask changes).
#
#         :param l1: A Level-1 data object
#         :return: A list of Level-1 data objects (subsets of polar ocean segments from input l1)
#         """
#
#         # Step: Filter small ocean segments
#         # NOTE: The objective is to remove any small marine regions (e.g. in fjords) that do not have any
#         #       reasonable chance of freeboard/ssh retrieval early on in the pre-processing.
#         if "ocean_mininum_size_nrecords" in self.cfg.polar_ocean:
#             logger.info("- filter ocean segments")
#             l1 = self.filter_small_ocean_segments(l1)
#
#         # Step: Extract Polar ocean segments from full orbit respecting the selected target hemisphere
#         logger.info("- extracting polar region subset(s)")
#         l1_list = self.trim_multiple_hemisphere_segment_to_polar_regions(l1)
#         logger.info(f"- extracted {len(l1_list)} polar region subset(s)")
#
#         # Step: Split the l1 segments at time discontinuities.
#         # NOTE: This step is optional. It requires the presence of the options branch `timestamp_discontinuities`
#         #       in the L1 pre-processor config file
#         if "timestamp_discontinuities" in self.cfg:
#             l1_list = self.split_at_time_discontinuities(l1_list)
#             logger.info(f"- split at time discontinuities -> {len(l1_list)} segments")
#
#         # Step: Trim the non-ocean parts of the subset (e.g. land, land-ice, ...)
#         # NOTE: Generally it can be assumed that the l1 object passed to this method contains polar ocean data.
#         #       But there tests before only include if there is ocean data and data above the polar latitude
#         #       threshold. It can therefore happen that trimming the non-ocean data leaves an empty Level-1 object.
#         #       In this case an empty list is returned.
#         logger.info("- trim outer non-ocean regions")
#         l1_trimmed_list = []
#         for l1 in l1_list:
#             l1_trimmed = self.trim_non_ocean_data(l1)
#             if l1_trimmed is not None:
#                 l1_trimmed_list.append(l1_trimmed)
#
#         # Step: Split the remaining subset at non-ocean parts.
#         # NOTE: There is no need to split the orbit at small features. See option `allow_nonocean_segment_nrecords`
#         #       in the l1p processor definition. But even if there are no segments to split, the output will always
#         #       be a list per requirements of the Level-1 pre-processor workflow.
#         l1_list = []
#         for l1 in l1_trimmed_list:
#             l1_splitted_list = self.split_at_large_non_ocean_segments(l1)
#             l1_list.extend(l1_splitted_list)
#
#         # All done, return the list of polar ocean segments
#         return l1_list
