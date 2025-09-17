# -*- coding: utf-8 -*-
"""
/***************************************************************************
 AgricultureProcessingProvider
                                 Processing Provider for Agriculture Mapping
                              -------------------
        begin                : 2025-01-25
        copyright            : (C) 2025 by Hiba Aamnous
        email                : hibaamnous@gmail.com
 ***************************************************************************/
"""

import os
from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .algorithms.satellite_acquisition import SatelliteAcquisitionAlgorithm
from .algorithms.data_preparation import DataPreparationAlgorithm
from .algorithms.model_training import ModelTrainingAlgorithm
from .algorithms.classification_mapping import ClassificationMappingAlgorithm
from .algorithms.topological_correction import TopologicalCorrectionAlgorithm


class AgricultureProcessingProvider(QgsProcessingProvider):
    """Processing Provider pour la cartographie des cultures"""

    def __init__(self):
        super().__init__()

    def loadAlgorithms(self):
        """Load all algorithms"""
        
        # 1. Acquisition d'images satellites
        self.addAlgorithm(SatelliteAcquisitionAlgorithm())
        
        # 2. Préparation des données d'entraînement
        self.addAlgorithm(DataPreparationAlgorithm())
        
        # 3. Entraînement des modèles ML
        self.addAlgorithm(ModelTrainingAlgorithm())
        
        # 4. Classification et cartographie
        self.addAlgorithm(ClassificationMappingAlgorithm())
        
        # 5. Correction topologique
        self.addAlgorithm(TopologicalCorrectionAlgorithm())

    def id(self):
        """Return provider id"""
        return 'agriculture_mapping'

    def name(self):
        """Return provider name"""
        return 'Cartographie des Cultures'

    def longName(self):
        """Return provider long name"""
        return 'Cartographie des Cultures - ML & Google Earth Engine'

    def icon(self):
        """Return provider icon"""
        plugin_dir = os.path.dirname(__file__)
        icon_path = os.path.join(plugin_dir, 'resources', 'icon.png')
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return QgsProcessingProvider.icon(self)

    def versionInfo(self):
        """Return version info"""
        return "2.0.0"

    def helpId(self):
        """Return help id"""
        return "agriculture_mapping_help"
