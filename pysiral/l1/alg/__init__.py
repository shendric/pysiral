# -*- coding: utf-8 -*-

"""
This module contains the basic functionality of Level-1 (pre-)processor items.

A processor item is a class that is allowed to modify the Level-1 data object
and can be called at different stages of the Level-1 pre-processor loop.
The classes are initialized when the Level-1 processor is started and
thus are allowed to map larger data sets to memory which are then applied
for each Level-1 data set.

Processing items can be anywhere in the pysiral namespace, but must
inherit `pysiral.l1preproc.procitems.L1PProcItem` and overwrite the
`apply(l1)` method, which receives the Level-1 data object as input.

Level-1 processor items are instanced from the Level-1 pre-processor
definition file (located in `pysiral/resources/pysiral-cfg/proc/l1`)
and require an entry according to this format:

```yaml
level1_preprocessor:

    ...

    options:

        ...

        processing_items:

            - label: <Label displayed in the log file>
              stage: <stage name (see below)>
              module_name: <the module where the class is located>
              class_name: <class name>
              options:
                  <custom option dictionary (can be nested)>
```

The value (stage name) supplied to `stage` defined whne the processing item
is run in the main loop of the Level-1 pre-processor. Following options
are currently implemented:

1. The processing item will be applied to the Level-1 data object, which
   is has the same extent as the source file (stage=`post_source_file`).
2. The processing item will be applied to a list of all polar ocean segments
   of a source file (stage=`post_ocean_segment_extraction`). This option is
   recommended for computation intensive processing items.
3. The processing item will be applied to the merged polar ocean segment
   (stage=`post_merge`)

"""

from loguru import logger
from typing import Dict
from pysiral import psrlcfg


class L1PProcItem(object):

    def __init_subclass__(cls) -> None:
        """
        Registers a class as Level-1 processor item

        :raises NotImplementedError: Subclass does not implement all required
            methods.

        :return: None
        """
        logger.debug(f"L1PProcItem: Register {cls.__name__}:{cls}")
        if "apply" not in cls.__dict__:
            raise NotImplementedError(f"{cls} does not implement required class apply")
        psrlcfg.class_registry.l1_proc_items[cls.__name__] = cls

    @classmethod
    def get_cls(cls, class_name: str, **options):
        """
        Get the initialized source data discovery class.

        :param class_name: A pysiral known Level-1 pre-processor class
        :param options: keyword arguments for the level-1 pre-processor items

        :raises KeyError: Invalid source_dataset_id.

        :return: Initialized source data discovery class
        """

        try:
            target_cls = psrlcfg.class_registry.l1_proc_items[class_name]
            return target_cls(**options)
        except KeyError as ke:
            msg = (
                f"Could not find Level-1 pre-processor item {class_name=} "
                f"[Available: {list(psrlcfg.class_registry.class_registry.keys())}]"
            )
            raise KeyError(msg) from ke
