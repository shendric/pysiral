Core Pysiral Concepts
=====================

Data Processing Levels
----------------------


Dataset ID's
------------



Radar Altimeter Input Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

``{platform}_{timeliness}_{agency}_{product_code}_{version}``

`Example: cryosat2_rep_esa_ice_00E`

Level-1 Pre-Processed Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

ID naming convention

``{platform}_{timeliness}_{version}``

`Example: cryosat2_rep_v1p0`

Pysiral higher level data products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ID naming convention:

``{processing_level}_{product_line}_{timeliness}_{platform}_{hemisphere}_{version}``

`Example: l2i_awi_rep_cryosat2_nh_v2p6`

Auxiliary Data
~~~~~~~~~~~~~~

ID naming convention:

``{auxiliary_type}_{dataset_name}_{version}``

`Example: sic_c3s_v3p0`

with:

- ``auxiliary_type``: The

**Auxiliary Dataset Types**

- ``ICECHARTS``: Sea ice classification from icecharts
- ``MDT``: Mean Dynamic Topography
- ``ML``: Models for Machine Learning Applications
- ``MSS``: Mean Sea Surface
- ``REGION``: Regional masks
- ``SNOW``: Data on snow thickness and/or density
- ``SIC``: Sea Ice Concentration
- ``SITYPE``: Sea Ice Type
-

