# -*- coding: utf-8 -*-
"""
Algorithme d'acquisition d'images satellites via Google Earth Engine
"""
import os
import ee
import time
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsWkbTypes, QgsProcessingAlgorithm, QgsProcessingParameterString,
    QgsProcessingParameterVectorLayer, QgsProcessingParameterDateTime,
    QgsProcessingParameterNumber, QgsProcessingParameterFileDestination,
    QgsProcessingParameterEnum, QgsProcessingException,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform
)

from ..utils.gee_utils import (
    ensure_gee_initialized,
    get_filtered_sentinel2_collection,
    process_indices_time_series,
    get_available_indices,
    download_sentinel2_image,
    calculate_vegetation_indices_s2,
    create_water_and_built_mask,
    debug_image_properties
)
from ..utils.validation_utils import validate_vector_layer, validate_numeric_parameter

class SatelliteAcquisitionAlgorithm(QgsProcessingAlgorithm):
    """Algorithme d'acquisition d'images satellites"""

    INPUT_AOI = 'INPUT_AOI'
    START_DATE = 'START_DATE'
    END_DATE = 'END_DATE'
    MAX_CLOUD_COVER = 'MAX_CLOUD_COVER'
    MAX_NODATA = 'MAX_NODATA'
    INDICES = 'INDICES'
    OUTPUT_CHART = 'OUTPUT_CHART'

    def initAlgorithm(self, config=None):
        """Initialisation des paramètres de l'algorithme"""
        info_text = (
            "Prérequis obligatoires :\n"
            "1. Compte Google Earth Engine activé\n"
            "2. API Earth Engine activée dans votre projet GCP\n"
            "3. Package 'earthengine-api' installé\n\n"
            "Installation des dépendances :\n"
            "Ouvrez OSGeo4W Shell et exécutez :\n"
            "python -m pip install earthengine-api"
        )

        self.addParameter(
            QgsProcessingParameterString(
                'INFO_BOX',
                self.tr("Informations importantes"),
                defaultValue=info_text,
                optional=False,
                multiLine=True
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                'GEE_PROJECT_ID',
                self.tr('ID de votre projet Google Cloud'),
                defaultValue='',
                optional=False
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                'MGRS_TILE',
                self.tr('Tuile MGRS (ex: 29SQU)'),
                defaultValue='',
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_AOI,
                self.tr("Zone d'étude (Polygone)")
            )
        )

        self.addParameter(
            QgsProcessingParameterDateTime(
                self.START_DATE,
                self.tr("Date de début")
            )
        )

        self.addParameter(
            QgsProcessingParameterDateTime(
                self.END_DATE,
                self.tr("Date de fin")
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_CLOUD_COVER,
                self.tr("Couverture nuageuse maximale (%)"),
                QgsProcessingParameterNumber.Double,
                defaultValue=20.0,
                optional=False
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_NODATA,
                self.tr("Données manquantes maximales (%)"),
                QgsProcessingParameterNumber.Double,
                defaultValue=20.0,
                optional=False
            )
        )
        available_indices = get_available_indices()
        indices_options = []

        for index_code, index_info in available_indices.items():
            option_text = f"{index_code} - {index_info['description']}"
            indices_options.append(option_text)

        self.addParameter(
            QgsProcessingParameterEnum(
                self.INDICES,
                self.tr('Indice de végétation à calculer'),
                options=indices_options,
                defaultValue=0
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Exécution de l'algorithme"""
        try:
            project_id = self.parameterAsString(parameters, 'GEE_PROJECT_ID', context)
            ensure_gee_initialized(feedback, project_id=project_id)
            time.sleep(2) 
            mgrs_tile = self.parameterAsString(parameters, 'MGRS_TILE', context)
            aoi_layer = self.parameterAsVectorLayer(parameters, self.INPUT_AOI, context)
            start_date = self.parameterAsDateTime(parameters, self.START_DATE, context).toString('yyyy-MM-dd')
            end_date = self.parameterAsDateTime(parameters, self.END_DATE, context).toString('yyyy-MM-dd')
            max_cloud = self.parameterAsDouble(parameters, self.MAX_CLOUD_COVER, context)
            max_nodata = self.parameterAsDouble(parameters, self.MAX_NODATA, context)
            indices_idx = self.parameterAsInt(parameters, self.INDICES, context)
            valid, errors = self.validate_parameters(aoi_layer, start_date, end_date, max_cloud, max_nodata, feedback)
            if not valid:
                raise QgsProcessingException("Erreurs de validation:\n" + "\n".join(errors))
            geometry = self.convert_geometry(aoi_layer, context).bounds()

            available_indices = get_available_indices()
            indices_list = list(available_indices.keys())
            selected_index = indices_list[indices_idx]
            indices_to_process = [selected_index]

            feedback.pushInfo(f"Traitement de l'indice: {selected_index}")

            filtered_collection = get_filtered_sentinel2_collection(
                geometry, start_date, end_date, max_cloud, max_nodata, feedback
            )
            debug_image_properties(filtered_collection, feedback)
            if mgrs_tile and mgrs_tile.strip() != '':
                filtered_collection = filtered_collection.filter(ee.Filter.eq('MGRS_TILE', mgrs_tile))
                feedback.pushInfo(f"Filtrage par tuile MGRS: {mgrs_tile}")

            index_img, index_masked = process_indices_time_series(
                filtered_collection, geometry, indices_to_process, apply_mask=True, feedback=feedback
            )
            if feedback:
                band_names = index_img.bandNames().getInfo()
                feedback.pushInfo(f"Bandes disponibles dans l'image d'indices: {band_names}")

            self.prepare_exports(filtered_collection, geometry, selected_index, feedback)

            feedback.pushInfo("✓ Traitement terminé avec succès")

            return {}

        except Exception as e:
            error_msg = f"Erreur lors du traitement: {str(e)}"
            if feedback:
                feedback.pushInfo(f"❌ {error_msg}")
            raise QgsProcessingException(error_msg)
    def validate_parameters(self, aoi_layer, start_date, end_date, max_cloud, max_nodata, feedback=None):
        """Validation complète des paramètres"""
        errors = []

        valid, layer_errors = validate_vector_layer(aoi_layer, feedback)
        if not valid:
            errors.extend(layer_errors)

        if start_date >= end_date:
            errors.append("La date de début doit être antérieure à la date de fin.")

        cloud_valid, cloud_msg = validate_numeric_parameter(max_cloud, 'Couverture nuageuse', 0, 100, feedback)
        if not cloud_valid:
            errors.append(cloud_msg)

        nodata_valid, nodata_msg = validate_numeric_parameter(max_nodata, 'Données manquantes', 0, 100, feedback)
        if not nodata_valid:
            errors.append(nodata_msg)

        return len(errors) == 0, errors

    def convert_geometry(self, aoi_layer, context, feedback=None):
        """Conversion de la géométrie QGIS vers EE avec la géométrie complète"""
        try:
            if feedback:
                feedback.pushInfo("Conversion de la géométrie complète...")

            if aoi_layer.featureCount() == 0:
                raise QgsProcessingException("La couche ne contient aucune entité")

            all_geometries = []
            
            for feature in aoi_layer.getFeatures():
                geom = feature.geometry()
                if not geom.isEmpty():
                    source_crs = aoi_layer.crs()
                    target_crs = QgsCoordinateReferenceSystem('EPSG:4326')
                    
                    if source_crs != target_crs:
                        transform = QgsCoordinateTransform(source_crs, target_crs, context.transformContext())
                        geom.transform(transform)

                    if geom.isMultipart():
                        for part in geom.asGeometryCollection():
                            all_geometries.append(part)
                    else:
                        all_geometries.append(geom)

            if not all_geometries:
                raise QgsProcessingException("Aucune géométrie valide trouvée")

            ee_geometries = []
            
            for geom in all_geometries:
                if geom.type() == QgsWkbTypes.PolygonGeometry:
                    polygon = geom.asPolygon()
                    for ring in polygon:
                        coords = [[p.x(), p.y()] for p in ring]
                        if len(coords) >= 3: 
                            ee_geometries.append(ee.Geometry.Polygon([coords]))
                elif geom.type() == QgsWkbTypes.LineGeometry:
                    line = geom.asPolyline()
                    coords = [[p.x(), p.y()] for p in line]
                    if len(coords) >= 2:
                        line_geom = ee.Geometry.LineString(coords)
                        ee_geometries.append(line_geom.buffer(10))
                elif geom.type() == QgsWkbTypes.PointGeometry:
                    point = geom.asPoint()
                    point_geom = ee.Geometry.Point([point.x(), point.y()])
                    ee_geometries.append(point_geom.buffer(10)) 

            if not ee_geometries:
                raise QgsProcessingException("Aucune géométrie convertible trouvée")

            if len(ee_geometries) == 1:
                final_geometry = ee_geometries[0]
            else:
                final_geometry = ee.Geometry.MultiPolygon([geom for geom in ee_geometries])

            if feedback:
                area = final_geometry.area(1).getInfo()
                feedback.pushInfo(f"✓ Zone d'étude convertie: {area:.2f} m²")
                feedback.pushInfo(f"✓ Nombre de polygones: {len(ee_geometries)}")

            return final_geometry

        except Exception as e:
            error_msg = f"Erreur de conversion géométrique: {str(e)}"
            if feedback:
                feedback.pushInfo(f"❌ {error_msg}")
            raise QgsProcessingException(error_msg)
    def prepare_exports(self, collection, geometry, index_name, feedback):
        """Préparation des exports - Version corrigée avec masquage"""
        try:
            n_images = collection.size().getInfo()
            if n_images == 0:
                raise QgsProcessingException("Aucune image disponible pour l'export")

            feedback.pushInfo(f"Création d'une image multibande à partir de {n_images} images...")

            def make_index_with_date_internal(img):
                img_with_index = calculate_vegetation_indices_s2(
                    img, geometry, [index_name], feedback=None
                )
                index_band = img_with_index.select(index_name.lower())

                image_id = img.get('system:index')
                band_name = ee.String(index_name).cat('_').cat(image_id)

                date_millis = img.get('system:time_start')
                return index_band.rename(band_name).set('system:time_start', date_millis)

            indices_collection = collection.map(make_index_with_date_internal)
            indices_multibande = indices_collection.toBands()
            water_mask, built_mask, full_mask = create_water_and_built_mask(collection, geometry, feedback)
            indices_masked = indices_multibande.updateMask(full_mask.Not())

            description_unmasked = f"{index_name}_FULL_TIMESERIES"
            description_masked = f"{index_name}_MASKED_TIMESERIES"

            download_sentinel2_image(indices_multibande, geometry, description_unmasked, feedback)
            download_sentinel2_image(indices_masked, geometry, description_masked, feedback)

            feedback.pushInfo("✅ Exports lancés : version normale et version masquée")

        except Exception as e:
            raise QgsProcessingException(f"Erreur lors de la préparation de l'export: {str(e)}")
        
    def name(self):
        return 'satellite_acquisition'

    def displayName(self):
        return self.tr('1 - Acquisition Données Satellite (GEE)')

    def group(self):
        return self.tr('Cartographie des Cultures')

    def groupId(self):
        return 'agriculture_mapping'

    def shortHelpString(self):
        return self.tr(
            "Acquisition automatique d'images satellites Sentinel-2 via Google Earth Engine.\n\n"
            "• Filtrage par couverture nuageuse et période temporelle\n"
            "• Calcul d'indices de végétation (NDVI, GNDVI, EVI, etc.)\n"
            "• Export automatique vers Google Drive\n"
            "• Génération de graphiques de qualité des données\n"
            "• Préparation pour analyse de cartographie des cultures\n\n"
            "Prérequis : Compte Google Earth Engine activé et authentifié."
        )


    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def icon(self):
        """Return algorithm icon"""
        from ..help_system import HelpSystem
        help_system = HelpSystem(os.path.dirname(os.path.dirname(__file__)))
        return help_system.get_algorithm_icon('satellite_acquisition')

    def helpUrl(self):
        """Return help URL"""
        from ..help_system import HelpSystem
        help_system = HelpSystem(os.path.dirname(os.path.dirname(__file__)))
        return help_system.get_help_url(self.name())

    def createInstance(self):
        return SatelliteAcquisitionAlgorithm()