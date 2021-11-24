# -*- coding: utf-8 -*-
"""

module for ingesting models from machine learning

Important Note:

    All mdt data handlers must be subclasses of pysiral.auxdata.AuxdataBaseClass in order to work
    for the Level-2 Processor. If the auxiliary class is based on a static dataset, this should be parsed
    in `__init__`.

    Please review the variables and properties in the parent class, as well as the correspodning config and
    support classes for grid track interpolation in the pysiral.auxdata module for additional guidance.

    The only other hard requirements is the presence of on specific method in order to be a valid subclass of
    AuxdataBaseClass:


        get_l2_track_vars(l2)

            This method will be called during the Level-2 processor. The argument is the Level-2 data object and
            the purpose of the method is to compute the auxilary variable(s) and associated uncertainty. These
            variable need to be registered using the `register_auxvar(id, name, value, uncertainty)` method of
            the base class. All MDT subclasses need to register at minimum the following variable:

            mean dynamic topography (relative to MSS):
                id: mdt
                name: mean_dynamic_topography

            e.g., this code line is mandatory for `get_l2_track_vars` (uncertainty can be None):

                # Register Variables
                self.register_auxvar("mdt", "mean_dynamic_topography", value, uncertainty)

"""
import re
from pathlib import Path
from typing import Iterable, Any, Union

import numpy as np
import xgboost as xgb

from pysiral.auxdata import AuxdataBaseClass
from pysiral.l2data import Level2Data
from pysiral.l1bdata import L1bdataNCFile

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


class RetrackerThresholdModel(AuxdataBaseClass):

    def __init__(self, *args: Iterable[Any], **kwargs: Iterable[Any]) -> None:
        """
        Initialiaze the class. This step includes establishing the model by parsing the
        model parameter file as specified in the Level-2 processor definition file.
        :param args:
        :param kwargs:
        """
        super(RetrackerThresholdModel, self).__init__(*args, **kwargs)

        # Query available model file
        suffixes = self.cfg.options.get("suffixes", [])
        model_files = self.get_available_model_files(self.cfg.local_repository, suffixes)

        # Retrieve requested model files
        model_id = self.cfg.options.get("model_id", None)
        if model_id is None:
            msg = f"Missing option `model_id` in auxiliary data configuration {self.cfg.options}"
            self.error.add_error("missing-option", msg)
            self.error.raise_on_error()
        model_filepath = [model_file for model_file in model_files if re.search(model_id, str(model_file))]

        # At this point there should only be one file
        if len(model_filepath) != 1:
            msg = f"No or multiple model files found for model_id = {model_id}: {self.cfg.local_repository}"
            self.error.add_error("ambigous-input", msg)
            self.error.raise_on_error()

        # Save input
        self.model_filepath = model_filepath[0]

        # NOTE: Danger zone - allowing multi-threading at which level?
        self.model_input = None
        self.model = xgb.Booster({'nthread': 1})
        self.model.load_model(str(self.model_filepath))

    def receive_l1p_input(self, l1p: 'L1bdataNCFile') -> None:
        """
        Optional method to add l1p variables to this class before `get_l2_track_vars()` is
        called. This method here grabs the necessary classifier parameter for the model
        prediction. The parameter names and their data group in the l1p data structure need to
        be specified in the Level-2 processor definition file as a list of [data_group, parameter_name],
        e.g.:

            parameter:
                - ["classifier", "leading_edge_width"]
                - ["classifier", "sigma0"]

        For this example the parameter list will be reorderd to a feature vector:

            [ [leading_edge_width[0], sigma0[0]], [leading_edge_width[1], sigma0[1]], ... ]

        :param l1p:
        :return:
        """

        # Retrieve required parameters for the model prediction
        parameter_list = self.cfg.options.get("parameter", [])
        if len(parameter_list) == 0:
            msg = f"Missing or empty option value parameter: {self.cfg.options}"
            self.error.add_error("missing-option", msg)
            self.error.raise_on_error()

        # Retrieve data from l1p data object
        model_input_list = []
        for data_group, parameter_name in parameter_list:
            model_input_list.append(l1p.get_parameter_by_name(data_group, parameter_name))

        # Reform the input data for the model prediction
        self.model_input = np.array(model_input_list).T

    def get_l2_track_vars(self, l2: 'Level2Data') -> None:
        """
        [Mandatory class method] Add the model prediction for the tfmra retracker threshold to the
        Level-2 data object. The model evaluation depends solely on waveform shape parameters that
        are not part of the Level-2 data object, because auxiliary data classes are called before the
        processor steps.
        :param l2:
        :return:
        """

        # Evaluate the model for this case
        prediction = self.model.predict(xgb.DMatrix(self.model_input))

        # Add prediction to the Level-2 data object
        var_id, var_name = self.cfg.options.get("output_parameter", ["tfmrathrs_ml", "tfmra_threshold_ml"])
        l2.set_auxiliary_parameter(var_id, var_name, prediction)

        # Remove the model input (safety measure)
        self.model_input = None

    def get_available_model_files(self, lookup_dir: Union[str, Path], suffixes: Iterable[str]) -> Iterable[Path]:
        """
        Check the avaiable model files
        :param lookup_dir:
        :param suffixes:
        :return: List of available model files
        """

        # Test if directory exists
        lookup_dir = Path(lookup_dir)
        if not lookup_dir.is_dir():
            msg = f"Directory for machine learned models does not exist: {lookup_dir}"
            self.error.add_error("directory-missing", msg)
            self.error.raise_on_error()

        # Find files
        model_files = []
        for suffix in suffixes:
            model_file_list = list(lookup_dir.rglob(f"*{suffix}"))
            model_files.extend(model_file_list)

        if len(model_files) == 0:
            msg = f"Did not find any machine learned files: {lookup_dir}/*{suffixes}"
            self.error.add_error("files-missing", msg)
            self.error.raise_on_error()

        return model_files
