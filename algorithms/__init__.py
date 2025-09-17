# -*- coding: utf-8 -*-
"""
Algorithmes de traitement pour le plugin Cartographie des Cultures
"""

from .satellite_acquisition import SatelliteAcquisitionAlgorithm
from .data_preparation import DataPreparationAlgorithm
from .model_training import ModelTrainingAlgorithm
from .classification_mapping import ClassificationMappingAlgorithm
from .topological_correction import TopologicalCorrectionAlgorithm

__all__ = [
    'SatelliteAcquisitionAlgorithm',
    'DataPreparationAlgorithm', 
    'ModelTrainingAlgorithm',
    'ClassificationMappingAlgorithm',
    'TopologicalCorrectionAlgorithm'
]