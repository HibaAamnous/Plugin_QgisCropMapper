# -*- coding: utf-8 -*-
"""
Utilitaires de validation pour le plugin
"""

import os
import numpy as np
import pandas as pd
from qgis.core import QgsRasterLayer


def validate_vector_layer(layer, feedback=None):
    errors = []
    
    if not layer:
        errors.append("Aucune couche sélectionnée")
        return False, errors
    if not layer.isValid():
        errors.append("La couche n'est pas valide")
    
    if layer.featureCount() == 0:
        errors.append("La couche ne contient aucune entité")
    for feature in layer.getFeatures():
        geom = feature.geometry()
        if geom.isNull() or geom.isEmpty():
            errors.append("La couche contient des géométries invalides")
            break
    
    return len(errors) == 0, errors

def validate_raster_layer(layer, feedback=None):
    """
    Valider une couche raster
    
    Args:
        layer: Couche raster QGIS
        feedback: Objet feedback pour les messages
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    try:
        if not layer:
            errors.append("Aucune couche raster sélectionnée")
            return False, errors
        
        if not isinstance(layer, QgsRasterLayer):
            errors.append("La couche n'est pas une couche raster")
            return False, errors
        
        if not layer.isValid():
            errors.append("La couche raster n'est pas valide")
            return False, errors
        
        width = layer.width()
        height = layer.height()
        if width == 0 or height == 0:
            errors.append("Dimensions du raster invalides")
        
        band_count = layer.bandCount()
        if band_count == 0:
            errors.append("Aucune bande dans le raster")

        crs = layer.crs()
        if not crs.isValid():
            errors.append("Système de coordonnées du raster invalide")
        
        source_path = layer.source()
        if not os.path.exists(source_path):
            errors.append(f"Fichier source introuvable: {source_path}")
        
        if feedback:
            feedback.pushInfo(f"Validation couche raster: {len(errors)} erreur(s)")
            feedback.pushInfo(f"  Dimensions: {width} x {height}")
            feedback.pushInfo(f"  Bandes: {band_count}")
            for error in errors:
                feedback.pushInfo(f"  - {error}")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"Erreur lors de la validation: {str(e)}")
        return False, errors


def validate_file_path(file_path, extensions=None, must_exist=False, feedback=None):
    """
    Valider un chemin de fichier
    
    Args:
        file_path: Chemin du fichier
        extensions: Liste des extensions autorisées (ex: ['.shp', '.geojson'])
        must_exist: Si True, le fichier doit exister
        feedback: Objet feedback pour les messages
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        if not file_path or file_path.strip() == '':
            return False, "Chemin de fichier vide"
        
        file_path = file_path.strip()
        if extensions:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in [ext.lower() for ext in extensions]:
                return False, f"Extension invalide. Extensions autorisées: {', '.join(extensions)}"
        
        if must_exist and not os.path.exists(file_path):
            return False, f"Fichier introuvable: {file_path}"
        if not must_exist:
            parent_dir = os.path.dirname(file_path)
            if parent_dir and not os.path.exists(parent_dir):
                return False, f"Dossier parent introuvable: {parent_dir}"
        
        if feedback:
            feedback.pushInfo(f"Validation chemin: {file_path} - ✓")
        
        return True, ""
        
    except Exception as e:
        error_msg = f"Erreur lors de la validation du chemin: {str(e)}"
        if feedback:
            feedback.pushInfo(error_msg)
        return False, error_msg

def validate_numeric_parameter(value, param_name, min_value=None, max_value=None, feedback=None):
    """
    Valider un paramètre numérique
    
    Args:
        value: Valeur à valider
        param_name: Nom du paramètre
        min_value: Valeur minimale autorisée
        max_value: Valeur maximale autorisée
        feedback: Objet feedback pour les messages
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        if value is None:
            return False, f"Paramètre '{param_name}' non défini"
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return False, f"Paramètre '{param_name}' doit être numérique"

        if min_value is not None and numeric_value < min_value:
            return False, f"Paramètre '{param_name}' doit être >= {min_value}"
        
        if max_value is not None and numeric_value > max_value:
            return False, f"Paramètre '{param_name}' doit être <= {max_value}"

        if np.isnan(numeric_value):
            return False, f"Paramètre '{param_name}' ne peut pas être NaN"
        
        if np.isinf(numeric_value):
            return False, f"Paramètre '{param_name}' ne peut pas être infini"
        
        if feedback:
            feedback.pushInfo(f"Paramètre '{param_name}' validé: {numeric_value}")
        
        return True, ""
        
    except Exception as e:
        error_msg = f"Erreur lors de la validation du paramètre '{param_name}': {str(e)}"
        if feedback:
            feedback.pushInfo(error_msg)
        return False, error_msg

def validate_acquisition_parameters(max_cloud_cover, max_nodata, start_date, end_date, feedback=None):
    """
    Valider les paramètres d'acquisition satellite
    
    Args:
        max_cloud_cover: Pourcentage maximum de couverture nuageuse
        max_nodata: Pourcentage maximum de données manquantes
        start_date: Date de début
        end_date: Date de fin
        feedback: Objet feedback pour les messages
    
    Returns:
        dict: Paramètres validés et éventuellement corrigés
    """
    warnings = []
    if max_cloud_cover > 50:
        warnings.append(f"⚠️ Couverture nuageuse élevée ({max_cloud_cover}%) - peu d'images disponibles")
        if feedback:
            feedback.pushInfo(f"⚠️ Couverture nuageuse élevée: {max_cloud_cover}%")

    if max_nodata > 50:
        warnings.append(f"⚠️ Seuil de données manquantes élevé ({max_nodata}%)")
        if feedback:
            feedback.pushInfo(f"⚠️ Seuil données manquantes élevé: {max_nodata}%")

    date_diff = (end_date - start_date).days
    if date_diff > 365:
        warnings.append("⚠️ Période très longue - traitement lent possible")
        if feedback:
            feedback.pushInfo(f"⚠️ Période de {date_diff} jours - traitement peut être lent")
    elif date_diff < 30:
        warnings.append("⚠️ Période courte - peu d'images disponibles")
        if feedback:
            feedback.pushInfo(f"⚠️ Période courte de {date_diff} jours")
    return {
        'max_cloud_cover': max_cloud_cover,
        'max_nodata': max_nodata,
        'start_date': start_date,
        'end_date': end_date,
        'warnings': warnings
    }


def validate_input_layer(layer, layer_type="vector", feedback=None):
    """
    Valider une couche d'entrée (vectorielle ou raster)
    
    Args:
        layer: Couche à valider
        layer_type: Type de couche ("vector" ou "raster")
        feedback: Objet feedback pour les messages
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    if layer_type == "vector":
        return validate_vector_layer(layer, feedback)
    elif layer_type == "raster":
        return validate_raster_layer(layer, feedback)
    else:
        return False, [f"Type de couche non supporté: {layer_type}"]
    

