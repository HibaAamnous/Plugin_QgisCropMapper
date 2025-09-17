# -*- coding: utf-8 -*-
"""
Algorithme de préparation des données d'entraînement
"""

import geopandas as gpd
import pandas as pd 
import os
import time
import random
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm, NULL,
    QgsProcessingParameterVectorLayer, QgsProcessingParameterField,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterNumber,
    QgsWkbTypes, QgsPointXY, QgsRaster,
    QgsProcessingParameterVectorDestination, QgsProcessingException,
    QgsRasterLayer, QgsGeometry, QgsCoordinateTransform, QgsProject
)
import numpy as np

class DataPreparationAlgorithm(QgsProcessingAlgorithm):
    """Algorithme de préparation des échantillons d'entraînement"""

    INPUT_PARCELS = 'INPUT_PARCELS'
    CULTURE_FIELD = 'CULTURE_FIELD'
    INPUT_IMAGE_RASTER = 'INPUT_IMAGE_RASTER'
    TRAIN_PERCENT = 'TRAIN_PERCENT'
    OUTPUT_SAMPLES = 'OUTPUT_SAMPLES'

    def initAlgorithm(self, config=None):
        """Initialize algorithm parameters"""
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_PARCELS,
                self.tr("Couche des Parcelles Agricoles"),
                [QgsProcessing.TypeVectorPolygon]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.CULTURE_FIELD,
                self.tr("Champ du Type de Culture"),
                parentLayerParameterName=self.INPUT_PARCELS,
                type=QgsProcessingParameterField.String
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_IMAGE_RASTER,
                self.tr("Image raster")
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TRAIN_PERCENT,
                self.tr("Pourcentage des données d'entraînement (%)"),
                QgsProcessingParameterNumber.Double,
                defaultValue=70.0,
                minValue=10.0,
                maxValue=90.0
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_SAMPLES,
                self.tr("Output des échantillons d'entraînement"),
                type=QgsProcessing.TypeVectorPoint,
                defaultValue=None,
                createByDefault=True,
                optional=False
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Process algorithm """

        feedback.pushInfo("⚠️ IMPORTANT: Assurez-vous que votre couche parcelles et votre raster sont dans le même système de coordonnées.")
        feedback.pushInfo("⚠️ Si ce n'est pas le cas, reprojetez-les d'abord dans QGIS.")
        parcels_layer = self.parameterAsVectorLayer(parameters, self.INPUT_PARCELS, context)
        culture_field_param = self.parameterAsString(parameters, self.CULTURE_FIELD, context)
        image_raster = self.parameterAsRasterLayer(parameters, self.INPUT_IMAGE_RASTER, context)
        image_raster_path = image_raster.source()
        train_percent = self.parameterAsDouble(parameters, self.TRAIN_PERCENT, context) / 100.0
        output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_SAMPLES, context)

        feedback.pushInfo("="*50)
        feedback.pushInfo("PRÉPARATION DES ÉCHANTILLONS D'ENTRAÎNEMENT")
        feedback.pushInfo("="*50)
        parcels_crs = parcels_layer.crs()
        raster_crs = image_raster.crs()

        if parcels_crs.authid() != raster_crs.authid():
            feedback.pushWarning(f"⚠️ ATTENTION: CRS différents détectés!")
            feedback.pushWarning(f"Parcelles: {parcels_crs.authid()}")
            feedback.pushWarning(f"Raster: {raster_crs.authid()}")
            feedback.pushWarning("Reprojetez vos données dans le même CRS avant de continuer.")

        try:
            feedback.pushInfo("🎯 Traitement direct du raster local...")

            samples_data, band_names = self.process_local_raster_direct(
                parcels_layer, image_raster_path, culture_field_param, train_percent, feedback
            )
            self.save_samples_to_shapefile(samples_data, band_names, culture_field_param, output_path, feedback)

            if context.willLoadLayerOnCompletion(output_path):
                details = context.layerToLoadOnCompletionDetails(output_path)
                details.name = "Échantillons d'entraînement"
                details.outputLayerName = "Échantillons d'entraînement"

        except Exception as e:
            error_msg = f"Erreur lors du traitement: {str(e)}"
            feedback.pushInfo(f"❌ {error_msg}")
            raise QgsProcessingException(error_msg)

        feedback.pushInfo("="*50)
        feedback.pushInfo("PRÉPARATION TERMINÉE AVEC SUCCÈS")
        feedback.pushInfo("="*50)

        return {self.OUTPUT_SAMPLES: output_path}

    def process_local_raster_direct(self, parcels_layer, raster_path, culture_field_param, train_percent, feedback):
        """Traitement du raster """
        feedback.pushInfo("Lecture du raster local...")
        raster_layer = QgsRasterLayer(raster_path, "temp_raster")
        if not raster_layer.isValid():
            raise QgsProcessingException(f"Impossible de charger le raster: {raster_path}")
        band_count = raster_layer.bandCount()
        if band_count == 0:
            raise QgsProcessingException("Le raster n'a aucune bande")
        
        band_names = [f"band_{i+1}" for i in range(band_count)]
        feedback.pushInfo(f"Raster avec {band_count} bandes détecté: {band_names}")
        samples_features = self.extract_points_from_parcels_with_grid(parcels_layer, raster_layer, culture_field_param, feedback)
        all_parcelle_ids = list(set([s['parcel_id'] for s in samples_features]))
        random.shuffle(all_parcelle_ids)
        num_train_parcels = int(len(all_parcelle_ids) * train_percent)
        parcelle_ids_train = set(all_parcelle_ids[:num_train_parcels])

        final_samples_data = []
        for i, sample_feature in enumerate(samples_features):
            parcel_id = sample_feature['parcel_id']
            is_train = 'True' if parcel_id in parcelle_ids_train else 'False'
            row_data = {
                'geometry': sample_feature['geometry'],
                'culture': sample_feature['culture'],
                'id_parcelle': parcel_id,
                'point_id': sample_feature['point_id'],
                'Train': is_train,
                'source_plugin': 'AgricultureMapping',
                'generation_date': time.strftime('%Y-%m-%d'),
                'train_percent': train_percent * 100
            }
            for band_name in band_names:
                row_data[band_name] = sample_feature['pixel_values'].get(band_name, 0.0)

            final_samples_data.append(row_data)

        if not final_samples_data:
            raise QgsProcessingException("Aucun échantillon généré")

        samples_gdf = gpd.GeoDataFrame(final_samples_data, geometry='geometry', crs=parcels_layer.crs().authid())

        return samples_gdf, band_names

    def extract_points_from_parcels_with_grid(self, parcels_layer, raster_layer, culture_field_param, feedback):
        """Extrait des points dans chaque parcelle avec une grille régulière de 10m"""
        points = []
        culture_field = culture_field_param
        fields = parcels_layer.fields()
        field_names = [field.name() for field in fields]
        
        if culture_field not in field_names:
            feedback.pushWarning(f"Champ '{culture_field}' non trouvé. Recherche automatique...")
            for field in fields:
                field_name = field.name().lower()
                if any(keyword in field_name for keyword in ['culture', 'crop', 'type', 'class']):
                    culture_field = field.name()
                    break
            
            if culture_field not in field_names:
                culture_field = fields[0].name() if fields else None
                feedback.pushWarning(f"Utilisation du champ: '{culture_field}'")

        if not culture_field:
            raise QgsProcessingException("Aucun champ de culture valide trouvé")

        feedback.pushInfo(f"Champ de culture utilisé: '{culture_field}'")
        feedback.pushInfo("Génération de points avec grille de 10m...")

        total_features = parcels_layer.featureCount()
        processed_features = 0

        for feature in parcels_layer.getFeatures():
            if feedback.isCanceled():
                break
                
            processed_features += 1
            if total_features > 0:
                feedback.setProgress(int(processed_features / total_features * 100))
            
            geom = feature.geometry()
            culture_type = feature.attribute(culture_field) if culture_field else "Unknown"
            if culture_type is None or culture_type == NULL: 
                culture_type = "Unknown"
            else:
                culture_type = str(culture_type)

            if geom and not geom.isEmpty() and geom.type() == QgsWkbTypes.PolygonGeometry:
                bbox = geom.boundingBox()
                xmin, ymin = bbox.xMinimum(), bbox.yMinimum()
                xmax, ymax = bbox.xMaximum(), bbox.yMaximum()

                width = xmax - xmin
                height = ymax - ymin
                num_x = int(width / 10) + 1
                num_y = int(height / 10) + 1
                
                point_count = 0
                for i in range(num_x):
                    for j in range(num_y):
                        x = xmin + i * 10 + 5  
                        y = ymin + j * 10 + 5
                        
                        point_geom = QgsGeometry.fromPointXY(QgsPointXY(x, y))
                        
                        if geom.contains(point_geom):
                            pixel_values = self.extract_pixel_values(point_geom, raster_layer, feedback)
                            
                            points.append({
                                'geometry': point_geom,
                                'culture': culture_type,
                                'parcel_id': feature.id(),
                                'point_id': f"{feature.id()}_{point_count}",
                                'pixel_values': pixel_values
                            })
                            point_count += 1
                if point_count == 0:
                    centroid = geom.centroid()
                    if centroid and not centroid.isEmpty():
                        pixel_values = self.extract_pixel_values(centroid, raster_layer, feedback)
                        
                        points.append({
                            'geometry': centroid,
                            'culture': culture_type,
                            'parcel_id': feature.id(),
                            'point_id': f"{feature.id()}_centroid",
                            'pixel_values': pixel_values
                        })

        feedback.pushInfo(f"Nombre total de points extraits: {len(points)}")
        cultures_found = set(point['culture'] for point in points if point['culture'])
        feedback.pushInfo(f"Cultures trouvées: {list(cultures_found)}")

        return points
    def extract_pixel_values(self, point_geometry, raster_layer, feedback=None):
        """Extrait les valeurs des pixels pour un point donné avec gestion robuste des erreurs."""
        pixel_values = {}
        
        if not point_geometry or point_geometry.isNull():
            if feedback:
                feedback.pushDebugInfo("Geometry is null or empty")
            return self._create_default_pixel_values(raster_layer)
        
        point = point_geometry.asPoint()
        provider = raster_layer.dataProvider()
        raster_extent = raster_layer.extent()
        if not raster_extent.contains(point):
            if feedback:
                feedback.pushDebugInfo(f"Point outside raster extent: {point.x()}, {point.y()}")
            return self._create_default_pixel_values(raster_layer)
        
        for band in range(1, raster_layer.bandCount() + 1):
            try:
                value, result = provider.sample(point, band)
                
                if result:
                    try:
                        float_value = float(value) if value is not None else 0.0
                        if np.isnan(float_value) or float_value == raster_layer.dataProvider().sourceNoDataValue(band):
                            float_value = 0.0
                    except (TypeError, ValueError):
                        float_value = 0.0
                        
                    pixel_values[f'band_{band}'] = float_value
                else:
                    pixel_values[f'band_{band}'] = 0.0
                    
            except Exception as e:
                if feedback:
                    feedback.pushDebugInfo(f"Error sampling band {band}: {str(e)}")
                pixel_values[f'band_{band}'] = 0.0
        
        return pixel_values

    def _create_default_pixel_values(self, raster_layer):
        """Crée des valeurs par défaut pour toutes les bandes du raster."""
        pixel_values = {}
        for band in range(1, raster_layer.bandCount() + 1):
            pixel_values[f'band_{band}'] = 0.0
        return pixel_values

    def save_samples_to_shapefile(self, samples_gdf, band_names, culture_field, output_path, feedback):
        """Sauvegarde les échantillons générés dans un shapefile """
        if samples_gdf.empty:
            raise QgsProcessingException("Aucun échantillon n'a été généré.")

        feedback.pushInfo(f"Sauvegarde de {len(samples_gdf)} échantillons dans {output_path}...")
        required_columns = ['id_parcelle', 'culture', 'Train', 'source_plugin', 'generation_date', 'train_percent']
        
        for col in required_columns:
            if col not in samples_gdf.columns:
                if col == 'train_percent':
                    samples_gdf[col] = 70.0 
                else:
                    samples_gdf[col] = ''

        for band_name in band_names:
            if band_name not in samples_gdf.columns:
                samples_gdf[band_name] = 0.0
        ordered_columns = ['id_parcelle', 'culture', 'Train', 'source_plugin', 'generation_date', 'train_percent'] + band_names
        available_columns = [col for col in ordered_columns if col in samples_gdf.columns]
        available_columns.insert(0, 'geometry')
        
        try:
            samples_gdf = samples_gdf[available_columns]
        except KeyError as e:
            raise QgsProcessingException(f"Erreur lors de la sélection des colonnes: {e}")

        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            for band_name in band_names:
                if band_name in samples_gdf.columns:
                    samples_gdf[band_name] = pd.to_numeric(samples_gdf[band_name], errors='coerce').fillna(0.0)

            samples_gdf['id_parcelle'] = samples_gdf['id_parcelle'].astype(str)
            samples_gdf['culture'] = samples_gdf['culture'].astype(str)
            samples_gdf['Train'] = samples_gdf['Train'].astype(str)

            samples_gdf.to_file(output_path, encoding='utf-8')
            feedback.pushInfo(f"✓ Échantillons sauvegardés avec succès en format Shapefile.")

            if os.path.exists(output_path):
                feedback.pushInfo(f"✓ Fichier créé: {output_path}")
                feedback.pushInfo(f"✓ Nombre de colonnes: {len(samples_gdf.columns)}")
                feedback.pushInfo(f"✓ Nombre d'échantillons: {len(samples_gdf)}")
            else:
                feedback.pushWarning(f"⚠️ Le fichier {output_path} n'a pas été créé")
                
        except Exception as e:
            error_msg = f"Erreur lors de la sauvegarde du shapefile: {str(e)}"
            feedback.pushInfo(f"❌ {error_msg}")
            raise QgsProcessingException(error_msg)

    def name(self):
        return 'data_preparation'

    def displayName(self):
        return self.tr('2 - Préparation des Échantillons ')

    def group(self):
        return self.tr('Cartographie des Cultures')

    def groupId(self):
        return 'agriculture_mapping'

    def shortHelpString(self):
        return self.tr("""
        <h3>Préparation des Échantillons d'Entraînement</h3>
        <p>Génère des échantillons d'entraînement en croisant les parcelles agricoles
        avec les données raster locales.</p>

        <h4>⚠️ IMPORTANT:</h4>
        <ul>
        <li><b>Reprojection requise:</b> Assurez-vous que vos parcelles et votre raster sont dans le même CRS</li>
        <li><b>Utilisez QGIS:</b> Reprojetez vos données si nécessaire avant d'utiliser cet algorithme</li>
        </ul>

        <h4>Fonctionnalités:</h4>
        <ul>
        <li><b>Échantillonnage spatial:</b> Extraction des valeurs pixel dans chaque parcelle avec grille de 10m</li>
        <li><b>Division train/test:</b> Séparation automatique par parcelle</li>
        <li><b>Multi-bandes:</b> Support des rasters multi-temporels</li>
        </ul>

        <h4>Entrées:</h4>
        <ul>
        <li><b>Parcelles:</b> Couche vectorielle des zones agricoles</li>
        <li><b>Champ culture:</b> Attribut contenant les types de cultures</li>
        <li><b>Raster:</b> Image multibande (ex: série temporelle d'indices)</li>
        <li><b>Pourcentage train:</b> Proportion des données d'entraînement (70% par défaut)</li>
        </ul>

        <h4>Sortie:</h4>
        <ul>
        <li><b>Échantillons:</b> Points avec valeurs spectrales et label train/test</li>
        </ul>
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def icon(self):
        """Return algorithm icon"""
        from ..help_system import HelpSystem
        help_system = HelpSystem(os.path.dirname(os.path.dirname(__file__)))
        return help_system.get_algorithm_icon('data_preparation')

    def helpUrl(self):
        """Return help URL"""
        from ..help_system import HelpSystem
        help_system = HelpSystem(os.path.dirname(os.path.dirname(__file__)))
        return help_system.get_help_url(self.name())
    
    def createInstance(self):
        return DataPreparationAlgorithm()