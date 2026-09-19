# -*- coding: utf-8 -*-
"""
Functions to help with dynamic loading of classes and modules
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import importlib
import multiprocessing
import pkgutil

from loguru import logger
from typing import Optional, Type, Tuple


def get_cls(
        module_name: str,
        class_name: str,
        relaxed: bool = True
) -> Tuple[Optional[Type], Optional[Exception]]:
    """
    Small helper function to dynamically load classes

    :param module_name: The name of the module to load (e.g. "pysiral.l2.l2preproc")
    :param class_name: The name of the class to load (e.g. "Level2Processor")
    :param relaxed:

    :return:
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        if relaxed:
            return None, e
        else:
            raise ImportError(f"Cannot load module: {module_name}") from e
    try:
        return getattr(module, class_name), None
    except AttributeError as e:
        if relaxed:
            return None, e
        else:
            raise NotImplementedError(f"Cannot load class: {module_name}.{class_name}") from e


def import_submodules(package: str, recursive: bool = True) -> dict[str, "Type"]:
    """
    Import all submodules of a module, recursively, including subpackages

    :param package: _package (name or actual module)
    :param recursive: Flag if _package is a submodule
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """

    from pysiral._exceptions import OptionalImportError

    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = f'{package.__name__}.{name}'
        try:
            results[full_name] = importlib.import_module(full_name)
        # Skip modules that cannot be imported due to missing optional dependencies
        except OptionalImportError:
            continue
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results


def set_psrl_cpu_count(cpu_count: int) -> int:
    """
    Set the pysiral-wide CPU count for multiprocessing to the pysiral _package
    configuration

    :param cpu_count: The number of CPU's to use

    :raises ValueError: cpu_count is not a positive integer
    """

    try:
        assert isinstance(cpu_count, int)
        assert cpu_count > 0
    except AssertionError as e:
        raise ValueError(
            f"specified number of CPU's ({cpu_count}) not a positive integer"
        ) from e
    cpu_count_mp = multiprocessing.cpu_count()
    if cpu_count > cpu_count_mp:
        logger.warning(f"Specified number of CPU's ({cpu_count}) > number of CPU's ({cpu_count_mp})")
    return cpu_count
