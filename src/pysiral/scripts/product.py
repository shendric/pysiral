# -*- coding: utf-8 -*-


import argparse

from typing import List, Optional
from dateperiods import DatePeriod
from pathlib import Path

from pysiral import PSRLCFG, set_psrl_cpu_count
from pysiral.scripts import l1preproc, l2proc, l3proc, l2preproc
from pysiral.scripts.parser_items import (
    ProcessingPeriod, ExcludeMonths, Hemisphere, PlatformID,
    ProductID, MultiProcesssingNumCores, UseMultiProcesssing,
    ProductProcessingLevel,
)


def product(
    processing_level: str,
    product_id: str,
    processing_period: DatePeriod,
    hemisphere: str,
    platform: Optional[str] = None,
    exclude_months: Optional[List[int]] = None,
    use_multiprocessing: bool = False,
    multiprocessing_num_cores: Optional[int] = None
) -> None:
    """
    pysiral product shortcut for the all pysiral processing levels (l1, l2, l3).

    The workflow for the processing levels can be different, therefore the
    input arguments are deferred to appropriate sub-functions.

    :param processing_level: The processing level to run (l1, l2, l3).
    :param product_id: The product ID to process.
    :param processing_period: The processing period to run the preprocessor for.
    :param exclude_months: List of months to exclude from processing (optional).
    :param hemisphere: Hemisphere to process data for.
    :param platform: Platform identifier (mandatory for product based processing).
    :param use_multiprocessing: Flag to enable multiprocessing (default is False).
    :param multiprocessing_num_cores: Number of cores to use for multiprocessing (optional).

    :return: None
    """

    kwargs = {
        "processing_level": processing_level,
        "product_id": product_id,
        "processing_period": processing_period,
        "exclude_months": exclude_months,
        "hemisphere": hemisphere,
        "platform": platform,
        "use_multiprocessing": use_multiprocessing,
        "multiprocessing_num_cores": multiprocessing_num_cores
    }
    if processing_level in ["l1", "l2"]:
        product_l1_l2(**kwargs)
    elif processing_level in ["l2p"]:
        kwargs.pop("use_multiprocessing")
        kwargs.pop("multiprocessing_num_cores")
        product_l2p(**kwargs)
    else:
        raise ValueError(f"Invalid processing level {processing_level}")


def product_l1_l2(
    processing_level: str,
    product_id: str,
    processing_period: DatePeriod,
    hemisphere: str,
    platform: Optional[str] = None,
    exclude_months: Optional[List[int]] = None,
    use_multiprocessing: bool = False,
    multiprocessing_num_cores: Optional[int] = None
) -> None:
    """
    Workflow script of the pysiral product shortcut for the all pysiral processing levels
    with single platform jobs

    :param processing_level: The processing level to run (l1, l2, l3).
    :param product_id: The product ID to process.
    :param processing_period: The processing period to run the preprocessor for.
    :param exclude_months: List of months to exclude from processing (optional).
    :param hemisphere: Hemisphere to process data for.
    :param platform: Platform identifier (mandatory for product based processing).
    :param use_multiprocessing: Flag to enable multiprocessing (default is False).
    :param multiprocessing_num_cores: Number of cores to use for multiprocessing (optional).

    :return: None
    """


    # Get the processor context for the given product ID and context variables
    context = {
        "product_id": product_id,
        "platform": platform,
        "hemisphere": hemisphere,
    }
    proc_settings = PSRLCFG.get_product_settings(processing_level, product_id, context)

    # Update settings with pass-through variables
    pass_through_keyw = {
        "processing_period": processing_period,
        "exclude_months": exclude_months,
        "use_multiprocessing": use_multiprocessing,
        "multiprocessing_num_cores": multiprocessing_num_cores,
    }
    proc_settings.update(pass_through_keyw)

    # Check if some settings file need to be resolved to a full path
    # TODO: This hack is needed due to some inconsistencies between `l1` and `l1p`
    data_level = {"l1p": "l1"}.get(processing_level, processing_level)
    for key in ["l1p_settings", "l2_settings", "l3_settings"]:
        if key in proc_settings and proc_settings[key] is not None:
            proc_settings[key] = PSRLCFG.get_settings_file("proc", data_level, proc_settings[key])

    # Check if some settings file need to be resolved to a full path
    output_processing_level = {"l2_outputs": "l2i", "l3_outputs": "l3"}
    for output_key in ["l2_outputs", "l3_outputs"]:
        if output_key in proc_settings and proc_settings[output_key] is not None:
            for i, output in enumerate(proc_settings[output_key]):
                proc_settings[output_key][i] = PSRLCFG.get_settings_file(
                    "output", output_processing_level[output_key], output
                )

    processors = {
        "l1p": l1preproc,
        "l2": l2proc,
        "l3": l3proc
    }
    # Run the appropriate processor with the settings obtained from the product definition
    processors[processing_level](**proc_settings)


# TODO: Rethink logic here (Maybe add platforms to override)
def product_l2p(
    processing_level: str,
    product_id: str,
    processing_period: DatePeriod,
    hemisphere: str,
    platform: Optional[str] = None,
    exclude_months: Optional[List[int]] = None,
) -> None:
    """
    Workflow script of the pysiral product shortcut for the all pysiral processing levels
    that will take multiple platforms

    :param processing_level: The processing level to run (l1, l2, l3).
    :param product_id: The product ID to process.
    :param processing_period: The processing period to run the preprocessor for.
    :param exclude_months: List of months to exclude from processing (optional).
    :param hemisphere: Hemisphere to process data for.
    :param platform: Platform identifier (mandatory for product based processing).

    :return: None
    """


    # Get the processor context for the given product ID and context variables
    context = {
        "product_id": product_id,
        "hemisphere": hemisphere,
    }
    proc_settings = PSRLCFG.get_product_settings(processing_level, product_id, context)


    # Update settings with pass-through variables
    pass_through_keyw = {
        "processing_period": processing_period,
        "exclude_months": exclude_months,
    }
    proc_settings.update(pass_through_keyw)

    # Check if some settings file need to be resolved to a full path
    # TODO: This hack is needed due to some inconsistencies between `l1` and `l1p`
    data_level = {"l1p": "l1"}.get(processing_level, processing_level)
    for key in ["l1p_settings", "l2_settings", "l3_settings"]:
        if key in proc_settings and proc_settings[key] is not None:
            proc_settings[key] = PSRLCFG.get_settings_file("proc", data_level, proc_settings[key])

    # Replace platforms with l2i input directories
    pysiral_product_dir = Path(PSRLCFG.local_machine["product_repository"])
    product_attributes = PSRLCFG.product.get_product_attributes(product_id)
    l2i_product_dirs = [
        pysiral_product_dir /
        product_attributes.product_line /
        product_attributes.record_type /
        platform /
        product_attributes.version /
        hemisphere /
        "l2i" for platform in proc_settings["platforms"]
    ]
    proc_settings["l2i_product_dirs"] = l2i_product_dirs
    proc_settings.pop("platforms")

    output_directory = (
        pysiral_product_dir /
        product_attributes.product_line /
        product_attributes.record_type /
        "multi-mission" /
        product_attributes.version /
        hemisphere /
        "l2p"
    )
    proc_settings["output_directory"] = output_directory

    # Check if some settings file need to be resolved to a full path
    output_processing_level = {"l2_outputs": "l2i", "l2p_outputs": "l2p", "l3_outputs": "l3"}
    for output_key in ["l2p_outputs"]:
        if output_key in proc_settings and proc_settings[output_key] is not None:
            for i, output in enumerate(proc_settings[output_key]):
                proc_settings[output_key][i] = PSRLCFG.get_settings_file(
                    "output", output_processing_level[output_key], output
                )

    processors = {
        "l2p": l2preproc,
    }
    # Run the appropriate processor with the settings obtained from the product definition
    processors[processing_level](**proc_settings)


class ProductScriptArguments(object):
    """
    """

    def __init__(self):
        """
        Command line parser class for the pysiral Level-2 Pre-Processor
        """
        self.parser = self.get_argument_parser()

    def get(self, args_list: Optional[List[str]] = None) -> "argparse.Namespace":
        args = self.parser.parse_args() if args_list is None else self.parser.parse_args(args_list)
        if args.multiprocessing_num_cores is not None:
            set_psrl_cpu_count(args.multiprocessing_num_cores)
        return args

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """
        Set up the command line argument parser for the Level-1 Pre-Processor.

        :return: The argument parser object.
        """

        # List of command line option required for the Level-1 pre-processor
        arg_item_list = [
            # Positional arguments
            ProductProcessingLevel().as_positional("processing_level"),
            ProductID().as_positional("product_id"),
            ProcessingPeriod().as_positional("processing_period"),
            # Optional arguments
            ExcludeMonths(),
            Hemisphere(),
            PlatformID(),
            UseMultiProcesssing(),
            MultiProcesssingNumCores()
        ]

        # create the parser
        parser = argparse.ArgumentParser(
            prog="pysiral product",
            description="""
            The product script is a shortcut to run the pysiral processing levels (l1, l2, l3) 
            for a given product ID. It requires less input parameters than the individual processing 
            level scripts and uses a product ID to determine the processing settings. 
            This script requires that a product definition files exist in `pysiral-cfg/product/{product_id}.yaml`
            """,
            epilog="For more information, see: https://pysiral.readthedocs.io",
            formatter_class=lambda prog: argparse.HelpFormatter(prog, width=96, indent_increment=4)  # noqa: E501
        )
        for arg_item in arg_item_list:
            arg_flags, args_dict = arg_item.get()
            parser.add_argument(*arg_flags, **args_dict)

        return parser
