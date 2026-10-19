# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from abc import ABC, abstractmethod

FILE_DISCOVERY_CLASS_REGISTRY: Dict[str, Type] = {}
DATA_LOADER_CLASS_REGISTRY: Dict[str, Type] = {}


class SourceDataLoader(ABC):
    """
    Abstract base class for source data loaders.
    """

    def __init_subclass__(cls):
        DATA_LOADER_CLASS_REGISTRY[cls.__name__] = cls

    @classmethod
    def get_cls(cls, class_name: str, *args, **kwargs):

        # Get the class
        try:
            target_cls = DATA_LOADER_CLASS_REGISTRY[class_name]
        except KeyError as ke:
            msg = (
                f"Could not find ObservationDataLoader class with {class_name=} "
                f"[Available: {list(DATA_LOADER_CLASS_REGISTRY.keys())}]"
            )
            raise KeyError(msg) from ke

        # Return the initialized class
        return target_cls(class_name, *args, **kwargs)

    @abstractmethod
    def get_l1(self, filepath: Path, **kwargs) -> Optional[Level1bData]:
        """
        Abstract method to load source data into Level 1 data object from a given file path.
        """
        pass


class FileDiscovery(ABC):
    """
    Abstract base class for file discovery.
    """

    def __init_subclass__(cls):
        FILE_DISCOVERY_CLASS_REGISTRY[cls.__name__] = cls

    @classmethod
    def get_cls(cls, class_name: str, *args, **kwargs):

        # Get the class
        try:
            target_cls = FILE_DISCOVERY_CLASS_REGISTRY[class_name]
        except KeyError as ke:
            msg = (
                f"Could not find FileDiscovery class with {class_name=} "
                f"[Available: {list(FILE_DISCOVERY_CLASS_REGISTRY.keys())}]"
            )
            raise KeyError(msg) from ke

        # Return the initialized class
        return target_cls(*args, **kwargs)

    @abstractmethod
    def get_l1(
            self,
            filepath: Path,
            polar_ocean_check: Optional[L1PreProcPolarOceanCheck] = None
    ) -> Optional[Level1bData]:
        """
        Abstract method to load source data into Level 1 data object from a given file path.
        """
        pass