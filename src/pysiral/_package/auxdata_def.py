# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


from datetime import timezone

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc

import yaml


class _AuxdataCatalogueItem(object):
    """
    Container for an auxiliary data item
    """

    def __init__(self, category, auxid, config_dict):
        """
        Data class to manage an auxiliary data set definition
        :param category:
        :param auxid:
        :param config_dict:
        """

        # Arguments
        self._category = category
        self._id = auxid
        self._config_dict = config_dict

    @property
    def id(self):
        return str(self._id)

    @property
    def category(self):
        return str(self._category)

    @property
    def keys(self):
        return self._config_dict.keys()

    @property
    def attrdict(self):
        return self._config_dict


class _AuxdataCatalogue(object):
    """
    Container for the content of the auxdata_def.yaml definition file
    for auxiliary data sets
    """

    def __init__(self, filepath):
        """
        Data container with query functionality for auxiliary data.
        :param filepath:
        """

        # Arguments
        self.filepath = filepath

        # Read contents
        with open(str(self.filepath)) as fh:
            self._yaml_content = yaml.safe_load(fh)

        # Create a catalogue of the content
        self.ctlg = {}
        for category, auxdata_items in self._yaml_content.items():
            self.ctlg[category] = {}
            for auxdata_id in auxdata_items:
                entry_dict = self._yaml_content[category][auxdata_id]
                item = _AuxdataCatalogueItem(category, auxdata_id, entry_dict)
                self.ctlg[category][auxdata_id] = item

    def get_category_items(self, category):
        """
        List all id's in a given category
        :param category:
        :return: list of ids
        """

        # Sanity check
        if category not in self.categories:
            raise ValueError(f'Invalid category: {str(category)} [{", ".join(self.categories)}]')

        # Return a sorted str list
        return sorted(self.ctlg[category].keys())

    def get_definition(self, category, auxid):
        """
        Retrieve the auxiliary data definition for a category and auxiliary data set id
        :param category: (str) Auxiliary data category (must be in self.categories)
        :param auxid: (str) The ID of the auxililary data set
        :return: AttrDict or None (if auxiliary dataset does not exist
        """

        # Check if valid category
        if category not in self.categories:
            return None

        # Extract & return the definition
        return self.ctlg[category].get(auxid, None)

    @property
    def categories(self):
        return self.ctlg.keys()

    @property
    def iter_keys(self):
        """
        List with two items per entry: (category, id)
        :return:
        """
        keys = []
        for category in self.categories:
            ids = self.get_category_items(category)
            keys.extend((category, auxid) for auxid in ids)
        return keys

    @property
    def items(self):
        """
        List with three items per entry: (category, id, catalogue_entry)
        :return:
        """
        keys = self.iter_keys
        return [(category, auxid, self.ctlg[category][auxid]) for category, auxid in keys]
