# -*- coding: utf-8 -*-


import argparse

from typing import List, Optional

from dateperiods import DatePeriod


from pysiral import PSRLCFG, set_psrl_cpu_count
from pysiral.scripts import l1preproc, l2proc, l3proc
from pysiral.scripts.parser_items import (
    ProcessingPeriod, ExcludeMonths, Hemisphere, PlatformID,
    ProductID, MultiProcesssingNumCores, UseMultiProcesssing,
    ProductProcessingLevel,
)


def product(
    processing_level: str,
    product_id: str,
    processing_period: DatePeriod,
    platform: str,
    hemisphere: str,
    exclude_months: Optional[List[int]] = None,
    use_multiprocessing: bool = False,
    multiprocessing_num_cores: Optional[int] = None
) -> None:
    """
    Workflow script of the pysiral product shortcut for the all pysiral processing levels (l1, l2, l3).

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
    for key in ["l1p_settings", "l2_settings", "l3_settings"]:
        if key in proc_settings and proc_settings[key] is not None:
            proc_settings[key] = PSRLCFG.get_settings_file("proc", processing_level, proc_settings[key])

    # Check if some settings file need to be resolved to a full path
    output_processing_level = {"l2_outputs": "l2i", "l3_outputs": "l3"}
    for output_key in ["l2_outputs", "l3_outputs"]:
        if output_key in proc_settings and proc_settings[output_key] is not None:
            for i, output in enumerate(proc_settings[output_key]):
                proc_settings[output_key][i] = PSRLCFG.get_settings_file(
                    "output", output_processing_level[output_key], output
                )

    processors = {
        "l1": l1preproc,
        "l2": l2proc,
        "l3": l3proc
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
