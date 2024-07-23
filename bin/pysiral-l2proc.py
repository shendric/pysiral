#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import glob
import re
import sys
import time

from pathlib import Path
from typing import Union, List, Optional, Literal
from datetime import timedelta

from dateperiods import DatePeriod
from loguru import logger

from pysiral import psrlcfg
from pysiral.core.cli import DefaultCommandLineArguments
from pysiral.core.datahandler import L1PDataHandler
from pysiral.core.errorhandler import ErrorStatus
from pysiral.l2 import Level2Processor, Level2ProductDefinition


def pysiral_l2proc_cli_wrapper() -> None:
    """
    The console scripts version of the pysiral-l1preprocessor scripts.
    It takes the required input from the command line with initial
    input processing and validation and starts the
    Level-1 pre-processor script.

    :return:
    """

    # Get the command line arguments
    cli = Level2ProcArgParser()
    cli.parse_command_line_arguments()

    # TODO: Implement ignore month
    target_period = DatePeriod(cli.args.start_date, cli.args.stop_date)

    pysiral_l2proc(
        cli.args.l1p_dataset_id,
        cli.args.l2_dataset_id,
        target_period,
        l2i_output_ids=cli.args.l2i_output_ids,
        overwrite_protection=cli.args.overwrite_protection
    )


def pysiral_l2proc(
        l1p_dataset_id: str,
        l2_dataset_id: str,
        period: DatePeriod,
        l2i_output_ids: Optional[Union[str, List[str]]] = None,
        overwrite_protection: Optional[bool] = False
) -> None:
    """

    Statement of Function
    ---------------------

    Starts the pysiral l2 processor generating a l2 dataset for a defined
    l1p dataset input dataset and period in monthly chunks. One Level-2 dataset
    is generated for each l1p dataset with identical coverage.

    Required Input Arguments
    ------------------------

    **L1P dataset id**: The identifier of the input data for the Level-2
    processors. pysiral will identify suitable l1p input files for the given period

    **L2p dataset id**: Identifier of the Level-2 dataset. Defines the Level-2
    processor algorithm definition and the (default) output format.

    **Processing Period**: A ``dateperiod.Dateperiod`` instance defining
    the processing period.


    Configuration Options
    ---------------------

    **L2 Output ID's**: Each l2 dataset is defined with a default output format, but the type
    of output can be overwritten with one or multiple out files using the ``l2i_output_ids``
    keyword argument. Please note that the content of ``l2i_output_ids` overwrites the
    default output.`

    :param l1p_dataset_id: Identifier of the l1p input dataset of type
        ``{platform}_{l1p_id}_{version}``. Pysiral will look in
        ``{pysiral_product_dir}/{platform}/{l1p_id}/{version}`` for data files.
    :param l2_dataset_id: Identifier of the l2 output dataset of type
        l2_{product_line}_{timeliness}_{platform}_{hemisphere}_{version}.
    :param period: The processing period definition.
    :param l2i_output_ids: (Optional): Overwrites the default l2 dataset
        output with (list of) output ids.t.
    :param overwrite_protection: (Optional) Boolean flag that is false be default.
        If set to true, pysiral will add a subdirectory with the processing data
        to the l2 output folder structure to avoid overwriting existing files.

    :raises ValueError: Incorrect l1p dataset id | Incorrect l2 dataset id
    """

    breakpoint()

    # # Collect job settings from pysiral configuration data and
    # # command line arguments
    # args = Level2ProcArgParser()
    #
    # # Parse and validate the command line arguments
    # args.parse_command_line_arguments()
    #
    # # Get confirmation for critical choices (if necessary)
    # args.critical_prompt_confirmation()
    #
    # # From here on there are two options
    # # a. Time range given -> Get l1bdata input with datahandler
    # # b. Predefined set of l1b input files
    # # Splitting into functions for clarity
    # if args.is_time_range_request:
    #     pysiral_l2proc_time_range_job(args)
    # else:
    #     pysiral_l2proc_l1b_predef_job(args)


def pysiral_l2proc_time_range_job(args):
    """ This is a Level-2 Processor job for a given time range """

    # Get start time of processor run
    t0 = time.process_time()

    # Get the product definition
    product_def = Level2ProductDefinition(args.run_tag,
                                          args.l2_settings_file,
                                          force_l2def_record_type=args.force_l2def_record_type)
    mission_id = product_def.l2def.metadata.platform
    hemisphere = product_def.l2def.metadata.hemisphere

    # Specifically add an output handler
    for l2_output in args.l2_output:
        product_def.add_output_definition(l2_output, overwrite_protection=args.overwrite_protection)

    # --- Get the period for the Level-2 Processor ---
    # Evaluate the input arguments
    period = DatePeriod(args.start, args.stop)

    # Clip the time range to the valid time range of the target platform
    period = period.intersect(psrlcfg.get_platform_period(mission_id))
    if period is None:
        msg = f"Invalid period definition ({args.start}-{args.stop}) for platform {mission_id}"
        raise ValueError(msg)

    # The Level-2 processor operates in monthly iterations
    # -> Break down the full period into monthly segments and
    #    filter specific month that should not be processed
    period_segments = period.get_segments("month", crop_to_period=True)
    if args.exclude_month is not None:
        period_segments.filter_month(args.exclude_month)

    # Prepare DataHandler
    l1b_data_handler = L1PDataHandler(
        mission_id,
        hemisphere,
        source_version=args.source_version,
        file_version=args.file_version
    )

    # Processor Initialization
    l2proc = Level2Processor(product_def)

    # Now loop over the month
    for time_range in period_segments:

        # Do some extra logging
        logger.info(f"Processing period: {time_range.label}")

        # Product Data Management
        if args.remove_old:
            for output_handler in product_def.output_handler:
                output_handler.remove_old(time_range)

        # Get input files
        l1b_files = l1b_data_handler.get_files_from_time_range(time_range)
        logger.info("Found %g files in %s" % (len(l1b_files), l1b_data_handler.last_directory))

        # Process the orbits
        l2proc.process_l1b_files(l1b_files)

    # All done
    t1 = time.process_time()
    seconds = int(t1 - t0)
    logger.info(f"Run completed in {str(timedelta(seconds=seconds))}")


def pysiral_l2proc_l1b_predef_job(args):
    """ A more simple Level-2 job with a predefined list of l1b data files """

    # Get start time of processor run
    t0 = time.process_time()

    # Get the product definition
    product_def = Level2ProductDefinition(
        args.run_tag,
        args.l2_settings_file,
        force_l2def_record_type=args.force_l2def_record_type
    )

    # Specifically add an output handler
    product_def.add_output_definition(
        args.l2_output,
        overwrite_protection=args.overwrite_protection
    )

    # Processor Initialization
    l2proc = Level2Processor(product_def)
    l2proc.process_l1b_files(args.l1b_predef_files)

    # All done
    t1 = time.process_time()
    seconds = int(t1 - t0)
    logger.info(f"Run completed in {str(timedelta(seconds=seconds))}")


class Level2ProcArgParser(object):

    def __init__(self):
        self.error = ErrorStatus()
        self.pysiral_config = psrlcfg
        self.args = None

    def parse_command_line_arguments(self):
        # use python module argparse to parse the command line arguments
        # (first validation of required options and data types)
        self.args = self.parser.parse_args()

        # Add additional check to make sure either `l1b-files` or
        # `start ` and `stop` are set
        # l1b_file_preset_is_set = self.args.l1b_files_preset is not None
        # start_and_stop_is_set = (
        #         self.args.start_date is not None and
        #         self.args.stop_date is not None
        # )

        # if l1b_file_preset_is_set and start_and_stop_is_set:
        #     self.parser.error("-start & -stop and -l1b-files are exclusive")
        #
        # if not l1b_file_preset_is_set and not start_and_stop_is_set:
        #     self.parser.error("either -start & -stop or -l1b-files required")

    def critical_prompt_confirmation(self):

        # Any confirmation prompts can be overridden by --no-critical-prompt
        no_prompt = self.args.no_critical_prompt

        # if --remove_old is set, all previous l1bdata files will be
        # erased for all month
        if self.args.remove_old and not no_prompt:
            message = "You have selected to remove all previous " + \
                      "l2 files for the requested period\n" + \
                      "(Note: use --no-critical-prompt to skip confirmation)\n" + \
                      "Enter \"YES\" to confirm and continue: "
            result = input(message)

            if result != "YES":
                sys.exit(1)

    @property
    def parser(self):
        # XXX: Move back to caller

        # Take the command line options from default settings
        # -> see config module for data types, destination variables, etc.
        clargs = DefaultCommandLineArguments()

        # List of command line option required for pre-processor
        # (argname, argtype (see config module), destination, required flag)
        options = [
            (("-l1", "--l1p-dataset-id",), "l1p-dataset-id", "l1p_dataset_id", True),
            (("-l2", "--l2-dataset-id", ), "l2-dataset-id", "l2_dataset_id", True),
            (("--start", ), "date", "start_date", False),
            (("--end",), "date", "stop_date", False),
            # ("-l1b-files", "l1b_files", "l1b_files_preset", False),
            # ("-exclude-month", "exclude-month", "exclude_month", False),
            (("--l2-output",), "l2-output", "l2_output", False),
            # ("--remove-old", "remove-old", "remove_old", False),
            # ("--force-l2def-record-type", "force-l2def-record-type", "force_l2def_record_type", False),
            # ("--no-critical-prompt", "no-critical-prompt", "no_critical_prompt", False),
            # ("--no-overwrite-protection", "no-overwrite-protection", "overwrite_protection", False),
            # ("--overwrite-protection", "overwrite-protection", "overwrite_protection", False)
        ]

        # create the parser
        parser = argparse.ArgumentParser()
        for option in options:
            argname, argtype, destination, required = option
            argparse_dict = clargs.get_argparse_dict(argtype, destination, required)
            parser.add_argument(argname, **argparse_dict)
        parser.set_defaults(overwrite_protection=False)

        return parser

    @property
    def arg_dict(self):
        """ Return the arguments as dictionary """
        return self.args.__dict__

    @property
    def start(self):
        return self.args.start_date

    @property
    def stop(self):
        return self.args.stop_date

    @property
    def run_tag(self):
        """ run_tag is a str or relative path that determines the output directory for
        the Level-2 processor. If the -run-tag option is not specified, the output
        directory will be the `product_repository` specification in `local_machine_def`
        with the l2 settings file basename as subfolder.

        One can however specify a custom string, or a relative path, with subfolders
        defined by using slashes or backslashes

        Examples:
            -run-tag cs2awi_v2p0_nrt
            -run-tag c3s/cdr/cryosat2/v1p0/nh
        """

        # Get from command line arguments (default: None)
        run_tag = self.args.run_tag

        # split the run-tag on potential path separators
        if run_tag is not None:
            run_tag = re.split(r'[\\|/]', run_tag)

        return run_tag

    @property
    def exclude_month(self):
        return self.args.exclude_month

    @property
    def overwrite_protection(self):
        return self.args.overwrite_protection

    @property
    def l2_settings_file(self):
        l2_settings = self.args.l2_settings
        filename = self.pysiral_config.get_settings_file("proc", "l2", l2_settings)
        if filename is not None:
            return filename
        msg = "Invalid l2 settings filename or id: %s\n" % l2_settings
        msg = msg + " \nRecognized Level-2 processor setting ids:\n"
        for l2_settings_id in self.pysiral_config.get_setting_ids("proc", "l2"):
            msg = f'{msg}  {l2_settings_id}' + "\n"
        self.error.add_error("invalid-l2-settings", msg)
        self.error.raise_on_error()

    @property
    def l1b_predef_files(self):
        return glob.glob(self.args.l1b_files_preset)

    @property
    def l2_output(self):
        filenames = []
        for l2_output in self.args.l2_output.split(";"):
            filename = self.pysiral_config.get_settings_file("output", "l2i", l2_output)

            if filename is None:
                msg = "Invalid l2 outputdef filename or id: %s\n" % l2_output
                msg = msg + " \nRecognized Level-2 output definitions ids:\n"
                l2_output_ids = self.pysiral_config.get_setting_ids("output", "l2i")
                for l2_output_id in l2_output_ids:
                    msg = f'{msg}    - {l2_output_id}' + "\n"
                self.error.add_error("invalid-l2-outputdef", msg)
                self.error.raise_on_error()
            else:
                filenames.append(filename)

        if not filenames:
            msg = "No valid output definition file found for argument: %s"
            msg %= (str(self.args.l3_output))
            self.error.add_error("invalid-outputdef", msg)
            self.error.raise_on_error()

        return filenames

    @property
    def force_l2def_record_type(self):
        return self.args.force_l2def_record_type

    @property
    def is_time_range_request(self):
        return self.args.l1b_files_preset is None

    @property
    def remove_old(self):
        return self.args.remove_old and not self.args.overwrite_protection


if __name__ == "__main__":
    pysiral_l2proc_cli_wrapper()
