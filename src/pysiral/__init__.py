# -*- coding: utf-8 -*-

"""
pysiral is the PYthon Sea Ice Radar ALtimetry toolbox
"""

__all__ = [
    "auxdata", "mission",
    "l1", "l2", "l3",
    "retracker",
    "psrlcfg",
    "import_submodules", "get_cls", "set_psrl_cpu_count",
    "__version__", "__git_version__", "__git_branch__", "__git_origin__"
]


from datetime import timezone


try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc

from pathlib import Path

# Does not import anything, just sets up the logger
# (before anything else)
import pysiral._logger  # isort: skip

# Read the version files
from pysiral._version import SOFTWARE_VERSION, GIT_VERSION, GIT_BRANCH, GIT_ORIGIN
# noinspection protected-member
from pysiral._package import PysiralPackageConfiguration, get_cls, import_submodules, set_psrl_cpu_count

PACKAGE_ROOT_DIR = Path(__file__).parent.resolve()

# Package Metadata
__version__ = SOFTWARE_VERSION
__git_version__ = GIT_VERSION
__git_branch__ = GIT_BRANCH
__git_origin__ = GIT_ORIGIN

# Create a _package configuration object as global variable
psrlcfg = PysiralPackageConfiguration()
