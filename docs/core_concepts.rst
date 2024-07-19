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

``{platform}_{source_id}_{timeliness}_{version}``

`Example: cryosat2_iceb0E_rep_v1p2`

Pysiral higher level data products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ID naming convention:

``{processing_level}_{product_line}_{platform}_{hemisphere}_{timeliness}_{version}``

`Example: l2i_awi_cryosat2_nh_rep_v2p6`

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
