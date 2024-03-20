# -*- coding: utf-8 -*-

"""
Configuration data model for the processor definition files
"""

from pathlib import Path
from collections import defaultdict
from typing import Dict, Literal, Optional, Union, List
from pydantic import BaseModel, PrivateAttr

from ._models import _YamlDefEntry
from pysiral.core.config import get_yaml_as_dict
from pysiral.core.dataset_ids import SourceDataID


class ProcDefCatalog(BaseModel):
    l1: Dict[str, _YamlDefEntry]
    l2: Dict[str, _YamlDefEntry]
    l3: Dict[str, _YamlDefEntry]
    _l1_ctlg: Dict = PrivateAttr(default={})

    def __init__(self, **data):
        """
        Catalogize the entries
        :param data:
        """
        super().__init__(**data)
        self._l1_ctlg = self._l1_get_supported_datasets()

        breakpoint()

    def _l1_get_supported_datasets(self) -> Dict:
        """
        Get a dict mapping of applicable l1 processor definitions for each
        source data set

        :return:
        """
        datasets_dict = defaultdict(dict)
        for l1p_name, l1p_def in self.l1.items():
            content_dict = get_yaml_as_dict(l1p_def.filepath)
            l1p_id = content_dict["pysiral_package_config"]["id"]
            for dataset_id in content_dict["pysiral_package_config"]["supported_source_datasets"]:
                datasets_dict[dataset_id][l1p_id] = l1p_def
        return datasets_dict

    def get_ids(self, processing_level: Literal["l1", "l2", "l3"]) -> List[str]:
        """
        Get a list of known processor settings id's for the specified processing level

        :param processing_level: One of l1, l2, l3

        :raises ValueError: Invalid processing level

        :return: A list of known processor id's
        """
        try:
            proc_level_cfg = getattr(self, processing_level)
        except AttributeError as ae:
            raise ValueError(f"Invalid processing level: {processing_level} [l1, l2, l3]") from ae

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

        :raises ValueError: Unknown dataset and l1p id

        :return: Filepath to Level-1 preprocessor config file
        """

        breakpoint()

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
