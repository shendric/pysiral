Glossary
========

.. glossary::

    Auxiliary Data 
        Any data set that is input to the geophysical retrieval or gridding that is not radar 
        altimeter sensor data (see :term:`Source Data`). Auxiliary data sets can be dynamic 
        (e.g. daily) or static fields. Examples are sea ice concentration or type products, 
        land/ocean or regions masks, data on snow on sea ice or mean sea surface products. 

    Level-1 Pre-Processor
        Dedicated processing step in pysiral that ingests source data and creates 
        orbit segments over the polar oceans with additional pre-computed parameters
        in a unified file format for all supported radar altimeter missions. 

    Level-2 Processor
        Dedicated processing step that performs the geophysical retrieval based on the output
        of the :term:`Level-1 Pre-Processor`. The coverage of the output is identical to the
        input data. 

    Level-2 Pre-Processor
        Dedicated processing step that ingests output from the :term:`Level-2 Processor` and
        filters and aggregates data and writes daily summary files with the same resolution
        of the Level-2 data. 

    Level-3 Processor
        Aggregates the output of the :term:`Level-2 Processor` on a spatio-temporal grid. 
        Also allows to compute statistics and additional parameters. 

    Mission
        A satellite mission that may consist of one or more satellite platforms. For example, 
        Sentinel-3A and Sentinel-3B are part of the Sentinel-3 mission. 

    Platform 
        A specific (satellite) platform. In pysiral, each platform is referenced by a unique
        platform identifier, which is by default the lower case name ([a-z0-9]) e.g. ``cryosat2`` or
        ``envisat``.

    Processing Level
        Data processing levels describe the state of data processing from lower  
        (close to the actual sensor data) to higher levels (geophysical retrievals). 

        +---------+-----------------------------------------------+
        | Level   | Description                                   |
        +=========+===============================================+
        | ``L0``  | Sensor raw data (not supported by pysiral)    |
        +---------+-----------------------------------------------+
        | ``L1B`` | Calibrated sensor data. Typical processing    |
        |         | level for :term:`Source Data`                 |
        +---------+-----------------------------------------------+
        | ``L1P`` | Pre-processed sensor data created by the      |
        |         | :term:`Level-1 Pre-Processor`                 |
        +---------+-----------------------------------------------+
        | ``L2i`` | Geophysical data at the same coverage and     |
        |         | resolutions of l1p data.                      |
        |         |                                               |
        |         | Output of the :term:`Level-2 Processor`       |
        +---------+-----------------------------------------------+
        | ``L2p`` | Aggregated and filtered l2i data, for example |
        |         | daily summary files only over sea ice         |
        |         |                                               |
        |         | Output of the :term:`Level-2 Pre-Processor`   |
        +---------+-----------------------------------------------+

    Sensor
        The name of the radar altimeter sensor. In pysiral, each sensor is referenced by a unique
        platform identifier, which is by default the lower case name e.g. ``siral`` for ``cryosat2`` or
        ``ra-2`` for ``envisat`` .

    Source Data 
        The term source data refers to calibrated radar altimeter data (waveforms) annotated with
        a land/ocean mask. geophysical range corrections for path delays in the atmosphere and 
        ionosphere as well as information from tide models. 

    Timeliness
        Defines the delay a data record is produced. Data from a specific platform/sensor
        is often delivered with more than one timeliness, and each of these products
        is its own :term:`Source Data` set. Datasets from satellites that are no longer 
        operational are classified as reprocessed. The table below gives an overview
        of frequently used timeliness codes and their typical delay. The actual delay 
        of indiviudal source data products may differ from the typical delay. 

        +---------+---------------------+---------------+---------+
        | Code    | Meaning             | Typical Delay | Alias   |
        +=========+=====================+===============+=========+
        | ``nrt`` | Near Real-Time      | < 2 days      | ``stc`` |
        +---------+---------------------+---------------+---------+
        | ``stc`` | Short Time Critical | < 2 day       | ``nrt`` |
        +---------+---------------------+---------------+---------+
        | ``rep`` | Reprocessed         | 1 month       | ``ntc`` |
        +---------+---------------------+---------------+---------+
        | ``ntc`` | Non Time Critical   | 1 month       | ``rep`` |
        +---------+---------------------+---------------+---------+



