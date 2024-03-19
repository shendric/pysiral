# -*- coding: utf-8 -*-

"""
Configuration data model for the processor definition files
"""

from pathlib import Path
from typing import Dict, Literal, Optional
from pydantic import BaseModel
from ._models import _YamlDefEntry


class ProcDefCatalog(BaseModel):
    l1: Dict[str, _YamlDefEntry]
    l2: Dict[str, _YamlDefEntry]
    l3: Dict[str, _YamlDefEntry]

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
