# -*- coding: utf-8 -*-

"""
Configuration data model for the processor definition files
"""

from pathlib import Path
from collections import defaultdict, namedtuple
from typing import Dict, Literal, Optional, Union, List
from pydantic import BaseModel, Field
from loguru import logger

from ._models import _YamlDefEntry
from pysiral.core.config import get_yaml_as_dict
from pysiral.core.dataset_ids import SourceDataID, L1PDataID


class ProcDefCatalog(BaseModel):
    """
    This class contains the catalog of processor definition files.
    Upon initialization the files in the pysiral configuration
    directory are added the catalog.

    """
    l1: Dict[str, _YamlDefEntry] = Field(
        description="ID's and filenames of Level-1 preprocessor definition files"
    )
    l2: Dict[str, _YamlDefEntry] = Field(
        description="ID's and filenames of Level-2 procesor definition files"
    )
    l3: Dict[str, _YamlDefEntry] = Field(
        description="ID's and filenames of Level-3 procesor definition files"
    )
    l1_ctlg: Dict = Field(
        description="Mapping of Level-1 preprocessor files to source data set id's",
        default=defaultdict(list)
    )

    l1p: Dict = Field(
        description="List of l1p data set ids (constructed from Level-1 preprocessor definition files)",
        default={}
    )

    def __init__(self, **data) -> None:
        """
        Create data catalogues after the

        :param data: The input to this class passed on to pydantic.BaseModel
        """
        super().__init__(**data)
        self._register_all_l1_procdefs()

    def register_l1procdef(self, filepath: Path) -> None:
        """
        Make a Level-1 pre-processor definition known to pysiral.
        The file will be parsed and linked to the supported datasets.

        :param filepath: Full filepath to yaml Level-1 pre-processor definition file
        """
        Entry = namedtuple("Entry", ["l1p_id", "filepath", "file_id"])
        content_dict = get_yaml_as_dict(filepath)
        l1p_id = content_dict["pysiral_package_config"]["l1p_id"]
        version = content_dict["pysiral_package_config"]["l1p_version"]
        for dataset_id in content_dict["pysiral_package_config"]["supported_source_datasets"]:
            self.l1_ctlg[dataset_id].append(Entry(l1p_id, filepath, filepath.stem))
            # foresee l1p dataset ids.
            sid = SourceDataID.from_str(dataset_id)
            l1_dataset_id = L1PDataID(
                platform=sid.platform_or_mission,
                source_dataset=l1p_id,
                timeliness=sid.timeliness,
                version=version
            )
            self.l1p[l1_dataset_id.version_str] = Entry(l1p_id, filepath, filepath.stem)

    def _register_all_l1_procdefs(self) -> None:
        """
        Register all l1 processor definitions in the pysiral package
        configuration.
        """
        for l1p_name, l1p_def in self.l1.items():
            self.register_l1procdef(l1p_def.filepath)

    def get_ids(self, processing_level: Literal["l1", "l1p", "l2", "l3"]) -> List[str]:
        """
        Get a list of known processor settings id's for the specified processing level

        :param processing_level: One of l1 (source), l1p, l2, l3

        :raises ValueError: Invalid processing level

        :return: A list of known processor id's
        """
        try:
            proc_level_cfg = getattr(self, processing_level)
        except AttributeError as ae:
            raise ValueError(f"Invalid processing level: {processing_level} [l1, l1p, l2, l3]") from ae
        return sorted(list(proc_level_cfg.keys()))

    def get_l1_from_dataset_id(
            self,
            dataset_id: Union[str, SourceDataID],
            l1p_id: str = None
    ) -> Path:
        """
        Get the Level-1 pre-processor definition file from the source dataset id and
        an optional l1p id. The l1p id only is required if more than one l1p processor
        definition supports the source data.

        :param dataset_id: The
        :param l1p_id:

        :raises ValueError: Source dataset id not known to pysiral and/or unknown l1p id

        :return: Filepath to Level-1 preprocessor config file
        """

        # Select all Level-1 pre-processor definitions that support source dataset
        l1p_proc_defs = self.l1_ctlg.get(dataset_id)
        if l1p_proc_defs is None:
            raise ValueError(f"\nUnknown {dataset_id=} [{self.l1_ctlg.keys()}]")
        n_procdefs = len(l1p_proc_defs)
        available_l1p_ids = [entry.l1p_id for entry in l1p_proc_defs]
        logger.info(f"Found {n_procdefs} Level-1 pre-processor definitions files for {dataset_id=}")

        # Validate l1p id (and get default value if not specified)
        l1p_id = self._validate_l1p_id(l1p_id, dataset_id, available_l1p_ids)
        idx = available_l1p_ids.index(l1p_id)
        return l1p_proc_defs[idx].filepath

    @staticmethod
    def _validate_l1p_id(
            l1p_id: Union[None, str],
            dataset_id: str,
            available_l1p_ids: List[str]
    ) -> str:
        """
        Validates l1p pre-processor definition id.

        :param l1p_id: The l1p id, is allowed to be None.
        :param available_l1p_ids: A list of available ids.

        :raises ValueError: l1p_is None with multiple available l1p id's or l1p_id
            is specified but not containd in `available_l1p_ids`

        :return: Validated l1p id
        """

        # Input validation: If more than one entry, `l1p_id` must not be None
        # Else it will be set to the only value
        if l1p_id is None:
            if len(available_l1p_ids) > 1:
                raise ValueError(
                    f"More than one Level-1 pre-processor definitions, but {l1p_id=}\n"
                    f"Select one of: {available_l1p_ids} (See `pysiral-l1preproc --help`)"
                )
            logger.info(f"Automatically setting l1p id to {available_l1p_ids[0]} for {dataset_id=}")
            return available_l1p_ids[0]

        if l1p_id not in available_l1p_ids:
            raise ValueError(
                f"\nUnknown Level-1 preprocessor id {l1p_id=} for {dataset_id=} {available_l1p_ids}"
            )
        logger.info(f"Using specfied {l1p_id=}")
        return l1p_id

    def get(
            self,
            proc_level: Literal["l1", "l2", "l3"],
            yaml_id: str,
            raise_if_none: bool = False
    ) -> Optional[Path]:
        """
        Return the file path for processor definition config base from the
        processing level and yaml file id.

        :param proc_level: A valid processor definition (l1, l2, l3)
        :param yaml_id: The id of the yaml file (filepath.setm)
        :param raise_if_none: Boolean flag defining action if yaml file does not exist

        :raises KeyError: `proc_level` not a valid processing level for processor
                definition files.

        :raises FileNotFoundError: Target yaml file does not exist and `raise_if_none=True`

        :return: filepath or None

        """
        if proc_level not in self.model_fields:
            if raise_if_none:
                raise KeyError(f"{proc_level=} not a valid processor level [l1, l2, l3]")
            return None

        yaml_def_entry = getattr(self, proc_level).get(yaml_id)
        if yaml_def_entry is None and raise_if_none:
            raise FileNotFoundError(f"No file found for {proc_level}:{yaml_id}")
        elif yaml_def_entry is None:
            return None
        return yaml_def_entry.filepath
