# -*- coding: utf-8 -*-

"""
Includes a debug map for L1 segments. This module requires the dev dependencies being installed.
"""


import contextlib
from typing import List

with contextlib.suppress(ImportError):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

from pysiral.l1 import Level1bData


__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


def l1p_debug_map(
        l1p_list: List["Level1bData"],
        title: str = None
) -> None:
    """
    Create an interactive map of l1p segment

    :param l1p_list:
    :param title:

    :return:
    """

    title = title if title is not None else ""
    proj = ccrs.PlateCarree()

    plt.figure(dpi=150)
    fig_manager = plt.get_current_fig_manager()
    try:
        fig_manager.window.showMaximized()
    except AttributeError:
        fig_manager.window.state('zoomed')
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.set_title(title)
    ax.coastlines(resolution='50m', color="0.25", linewidth=0.25, zorder=201)
    ax.add_feature(cfeature.OCEAN, color="#D0CFD4", zorder=150)
    ax.add_feature(cfeature.LAND, color="#EAEAEA", zorder=200)

    for i, l1p in enumerate(l1p_list):
        ax.scatter(l1p.time_orbit.longitude, l1p.time_orbit.latitude,
                   s=1, zorder=300, linewidths=0.0,
                   transform=proj
                   )
        ax.scatter(l1p.time_orbit.longitude[0], l1p.time_orbit.latitude[0],
                   s=10, zorder=300, linewidths=0.5, color="none", edgecolors="black",
                   transform=proj
                   )

        ax.annotate(f"{i+1}",
                    xy=(l1p.time_orbit.longitude[0], l1p.time_orbit.latitude[0]),
                    xycoords=proj._as_mpl_transform(ax), zorder=300,
                    xytext=(10, 10), textcoords="offset pixels",
                    fontsize=6
                    )
    plt.show()
