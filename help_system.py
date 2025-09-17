# -*- coding: utf-8 -*-
"""
Système d'aide pour le plugin de Cartographie des Cultures
Développé par Hiba Aamnous - Pôle Digital de l'Agriculture
"""

import os
import webbrowser
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtWidgets import QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsMessageLog, Qgis

class HelpSystem:
    """Gestionnaire du système d'aide du plugin"""
    
    def __init__(self, plugin_dir):
        self.plugin_dir = plugin_dir
        self.help_dir = os.path.join(plugin_dir, 'help')
        self.resources_dir = os.path.join(plugin_dir, 'resources')
        self.icons_dir = os.path.join(self.resources_dir, 'icons')
    
    def show_main_help(self):
        """Affiche la page d'aide principale"""
        help_file = os.path.join(self.help_dir, 'index.html')
        if os.path.exists(help_file):
            file_url = QUrl.fromLocalFile(help_file).toString()
            webbrowser.open(file_url)
        else:
            QMessageBox.warning(
                None,
                "Aide non disponible",
                "Le fichier d'aide principal est introuvable."
            )
    
    def validate_help_files(self):
        missing_files = []
        
        if not os.path.exists(os.path.join(self.help_dir, 'index.html')):
            missing_files.append('index.html')
        
        for help_file in self.help_files.values():
            if not os.path.exists(os.path.join(self.help_dir, help_file)):
                missing_files.append(help_file)
        
        return missing_files

    def show_algorithm_help(self, algorithm_name):
        """Affiche l'aide spécifique à un algorithme"""
        help_files = {
            'satellite_acquisition': 'Acquisition_Satellites_help.html',
            'data_preparation': 'data_preparation_help.html', 
            'model_training': 'model_training_help.html',
            'classification_mapping': 'classification_help.html',
            'topological_correction': 'topology_help.html'
        }

        if algorithm_name in help_files:
            help_file = os.path.join(self.help_dir, help_files[algorithm_name])
            if os.path.exists(help_file):
                file_url = QUrl.fromLocalFile(help_file).toString()
                webbrowser.open(file_url)
                QgsMessageLog.logMessage(
                    f"Documentation ouverte pour l'algorithme : {algorithm_name}",
                    "QgisCropMapper",
                    Qgis.Info
                )
                return True
            else:
                QMessageBox.warning(
                    None,
                    "Documentation non disponible",
                    f"Le fichier d'aide pour l'algorithme '{algorithm_name}' est introuvable.\nVérifiez l'installation du plugin."
                )
                return False
        else:
            QgsMessageLog.logMessage(
                f"Algorithme non reconnu: {algorithm_name}. Ouverture de l'aide générale",
                "QgisCropMapper", 
                Qgis.Warning
            )
            self.show_main_help()
            return False
    
    def get_algorithm_icon(self, algorithm_name):
        """Retourne l'icône pour un algorithme spécifique"""
        icon_files = {
            'satellite_acquisition': 'satellite.svg',
            'data_preparation': 'data_prep.svg',
            'model_training': 'training.svg',
            'classification_mapping': 'classification.svg',
            'topological_correction': 'topology.svg'
        }
        
        if algorithm_name in icon_files:
            icon_path = os.path.join(self.icons_dir, icon_files[algorithm_name])
            if os.path.exists(icon_path):
                return QIcon(icon_path)
    
        default_icon = os.path.join(self.resources_dir, 'icon.png')
        if os.path.exists(default_icon):
            return QIcon(default_icon)
        
        return QIcon()
    
    def get_help_url(self, algorithm_name):
        """Retourne l'URL d'aide pour un algorithme"""
        help_files = {
            'satellite_acquisition': 'Acquisition_Satellites_help.html',
            'data_preparation': 'data_preparation_help.html',
            'model_training': 'model_training_help.html',
            'classification_mapping': 'classification_help.html',
            'topological_correction': 'topology_help.html'
        }
        
        if algorithm_name in help_files:
            help_file = os.path.join(self.help_dir, help_files[algorithm_name])
            if os.path.exists(help_file):
                return QUrl.fromLocalFile(help_file).toString()
        
        main_help = os.path.join(self.help_dir, 'index.html')
        if os.path.exists(main_help):
            return QUrl.fromLocalFile(main_help).toString()
        
        return ""
    
