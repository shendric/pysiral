# -*- coding: utf-8 -*-

"""
pysiral is the PYthon Sea Ice Radar ALtimetry toolbox
"""

import importlib
import logging
import multiprocessing
import os
import pkgutil
import subprocess
import sys
import warnings
from typing import Dict
from pathlib import Path
from loguru import logger
from datetime import datetime
from dateutil.relativedelta import relativedelta

from _package_config import PysiralPackageConfiguration

__all__ = ["auxdata", "cryosat2", "envisat", "ers", "sentinel3",
           "filter", "frb", "grid", "_package_config", "core",
           "l1data", "l1preproc", "l2data", "l2preproc", "l2proc", "l3proc",
           "mask", "retracker",
           "sit", "surface", "waveform", "psrlcfg", "import_submodules", "get_cls",
           "set_psrl_cpu_count", "InterceptHandler", "__version__"]


warnings.filterwarnings("ignore")

# Set standard logger format
logger.remove()


def get_duration(elapsed_seconds: int, fmt_str: str = "%H:%M:%S") -> str:
    # Example time
    datum = datetime(1900, 1, 1)
    duration = datum + relativedelta(seconds=elapsed_seconds)
    return duration.strftime(fmt_str)


def logger_format(record: Dict) -> str:
    """
    Formatter function for loguru (Normal mode

    :param record:

    :return: logger string
    """
    elapsed_timestr = get_duration(record["elapsed"].total_seconds())
    fmt_str = (
        '<green>{time:YYYY-MM-DD HH:mm:ss.SS} - {elapsed_timestr}</green>  | '
        '<level>{level:<8}</level> | '
        '<level>{message}</level>\n'
    )
    return fmt_str.format(elapsed_timestr=elapsed_timestr, **record)


def logger_format_debug(record: Dict) -> str:
    elapsed_timestr = get_duration(record["elapsed"].total_seconds())
    if record["function"] == "<module>":
        record["function"] = ""
    else:
        record["function"] = "." + record["function"]
    code_line = "{name}{function}:L{line}".format(**record)
    print(code_line)
    fmt_str = (
        '<green>{time:YYYY-MM-DD HH:mm:ss.SS} - {elapsed_timestr}</green> | '
        '<level>{level: <8}</level> | '
        '<cyan>{code_line}</cyan> : '
        '<level>{message}</level>\n'
    )
    return fmt_str.format(elapsed_timestr=elapsed_timestr, code_line=code_line, **record)


PYSIRAL_DEBUG_MODE = "PYSIRAL_DEBUG_MODE" in os.environ
LOG_LEVEL = "DEBUG" if PYSIRAL_DEBUG_MODE else "INFO"
logger_format_func = logger_format_debug if PYSIRAL_DEBUG_MODE else logger_format
logger.add(sys.stderr, format=logger_format_func, enqueue=True, level=LOG_LEVEL)


# TODO: Make obsolete by porting sampy into pysiral
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# Get git version (allows tracing of the exact commit)
# TODO: This only works when the code is in a git repository (and not as installed python package)
try:
    __software_version__ = subprocess.check_output(["git", "log", "--pretty=format:%H", "-n", "1"])
    __software_version__ = __software_version__.strip().decode()
except (FileNotFoundError, subprocess.CalledProcessError):
    __software_version__ = None


# Get version from VERSION in package root
PACKAGE_ROOT_DIR = Path(__file__).absolute().parent
VERSION_FILE_PATH = PACKAGE_ROOT_DIR / "VERSION"
try:
    version_file = open(str(VERSION_FILE_PATH))
    with version_file as f:
        version = f.read().strip()
except IOError:
    sys.exit(f'Cannot find VERSION file in package (expected: {PACKAGE_ROOT_DIR / "VERSION"}')


# Package Metadata
__version__ = version
__author__ = "Stefan Hendricks"
__author_email__ = "stefan.hendricks@awi.de"


# Create a package configuration object as global variable
psrlcfg = PysiralPackageConfiguration(PACKAGE_ROOT_DIR, __version__)


def get_cls(module_name, class_name, relaxed=True):
    """ Small helper function to dynamically load classes"""
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


def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :param recursive: Flag if package is a submodule
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = f'{package.__name__}.{name}'
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results |= import_submodules(full_name)
    return results


def set_psrl_cpu_count(cpu_count: int) -> None:
    """
    Set the pysiral-wide CPU count for multiprocessing to the pysiral package
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
    psrlcfg.CPU_COUNT = cpu_count


# Recursively import all pysiral submodules.
# The reason for this is to register all relevant classes
# in the pysiral package configuration via subclassing
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    import time
    t0 = time.time()
    _module = loader.find_module(module_name).load_module(module_name)
    t1 = time.time()
    logger.debug(f"Imported {module_name} in {t1-t0:.3f} seconds")
    globals()[module_name] = _module
