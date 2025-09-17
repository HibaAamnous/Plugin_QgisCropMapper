# -*- coding: utf-8 -*-
"""
Algorithme de classification et cartographie des cultures
"""
import errno
import os
import joblib
import numpy as np
from sklearn.impute import SimpleImputer  
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterFile,
    QgsProcessingParameterRasterDestination, QgsProcessingParameterFileDestination,
    QgsProcessingParameterBoolean, QgsProcessingException,
    QgsRasterLayer, QgsProject
)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    gdal = None
    rasterio = None

class ClassificationMappingAlgorithm(QgsProcessingAlgorithm):
    """Algorithme de classification et cartographie"""
    
    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_MODEL = 'INPUT_MODEL'
    INPUT_SCALER = 'INPUT_SCALER'
    INPUT_LABEL_ENCODER = 'INPUT_LABEL_ENCODER'
    OUTPUT_CLASSIFICATION = 'OUTPUT_CLASSIFICATION'
    OUTPUT_PROBABILITY = 'OUTPUT_PROBABILITY'
    GENERATE_MAP = 'GENERATE_MAP'
    OUTPUT_MAP = 'OUTPUT_MAP'

    def initAlgorithm(self, config=None):
        """Initialize algorithm parameters"""
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr('Image satellite à classifier')
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_MODEL,
                self.tr('Modèle ML entraîné'),
                extension='pkl'
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_SCALER,
                self.tr('Fichier Scaler'),
                extension='pkl'
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_LABEL_ENCODER,
                self.tr('Fichier Label Encoder'),
                extension='pkl'
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_CLASSIFICATION,
                self.tr('Carte de classification des cultures')
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_PROBABILITY,
                self.tr('Carte des probabilités maximales'),
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.GENERATE_MAP,
                self.tr('Générer une carte thématique (PNG/PDF)'),
                defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_MAP,
                self.tr('Carte thématique'),
                fileFilter='PNG files (*.png);;PDF files (*.pdf)',
                optional=True
            )
        )
    def processAlgorithm(self, parameters, context, feedback):
        """Process algorithm"""
        if not rasterio:
            raise QgsProcessingException(
                "Ce module nécessite rasterio. Installez-le avec:python.exe -m pip install rasterio"
            )
        input_raster = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        model_path = self.parameterAsString(parameters, self.INPUT_MODEL, context)
        scaler_path = self.parameterAsString(parameters, self.INPUT_SCALER, context)
        label_encoder_path = self.parameterAsString(parameters, self.INPUT_LABEL_ENCODER, context)
        output_classification = self.parameterAsOutputLayer(parameters, self.OUTPUT_CLASSIFICATION, context)
        output_probability = self.parameterAsOutputLayer(parameters, self.OUTPUT_PROBABILITY, context)
        generate_map = self.parameterAsBool(parameters, self.GENERATE_MAP, context)
        output_map = self.parameterAsFileOutput(parameters, self.OUTPUT_MAP, context)

        for path in [output_classification, output_probability]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    raise QgsProcessingException(
                        f"Impossible de supprimer le fichier existant {path}. "
                        "Veuillez fermer le fichier dans QGIS et dans toute autre application."
                    )
                except OSError as e:
                    if e.errno == errno.EACCES:
                        raise QgsProcessingException(
                            f"Permission denied pour {path}. "
                            "Veuillez vérifier les permissions du dossier."
                        )       
        feedback.pushInfo("="*60)
        feedback.pushInfo("CLASSIFICATION ET CARTOGRAPHIE DES CULTURES")
        feedback.pushInfo("="*60)
        
        try:
            feedback.pushInfo("Chargement du modèle, scaler et label encoder...")
            for path in [model_path, scaler_path, label_encoder_path]:
                if not os.path.exists(path):
                    raise QgsProcessingException(f"Fichier introuvable: {path}")
            try:
                loaded_data = joblib.load(model_path)

                if isinstance(loaded_data, dict):
                    model = loaded_data['model']
                    feature_names = loaded_data.get('feature_names', [])
                    class_names_from_model = loaded_data.get('class_names', [])
                else:
                    model = loaded_data
                    feature_names = []
                    class_names_from_model = []
                
                scaler = joblib.load(scaler_path)
                label_encoder = joblib.load(label_encoder_path)
            except Exception as e:
                raise QgsProcessingException(f"Erreur de chargement des fichiers: {str(e)}")

            class_names = label_encoder.classes_
            if class_names_from_model and not np.array_equal(class_names, class_names_from_model):
                feedback.pushWarning("⚠️ Incohérence entre class_names du model et label_encoder. Vérifiez les fichiers.")
            model_name = type(model).__name__
            feedback.pushInfo(f"✓ Modèle chargé: {model_name}")
            feedback.pushInfo(f"✓ Classes: {', '.join(class_names)}")
            feedback.pushInfo("Lecture de l'image satellite...")
            raster_path = input_raster.source()
            
            with rasterio.open(raster_path) as src:
                width, height = src.width, src.height
                transform = src.transform
                crs = src.crs
                n_bands = src.count
                if feature_names and len(feature_names) != n_bands:
                    feedback.pushWarning(f"⚠️ Nombre de bandes dans l'image ({n_bands}) ne match pas les features du model ({len(feature_names)}). Cela peut causer des erreurs.")
                block_height, block_width = src.block_shapes[0]
                
                feedback.pushInfo(f"✓ Dimensions: {width} x {height} pixels")
                feedback.pushInfo(f"✓ Bandes: {n_bands}")
                feedback.pushInfo(f"✓ CRS: {crs}")
                feedback.pushInfo(f"✓ Taille des blocs: {block_width} x {block_height}")
                if hasattr(scaler, 'n_features_in_') and n_bands != scaler.n_features_in_:
                    feedback.pushWarning(
                        f"⚠️ Attention: {n_bands} bandes dans l'image vs "
                        f"{scaler.n_features_in_} caractéristiques dans le scaler"
                    )
                imputer = SimpleImputer(strategy='constant', fill_value=0)
                imputer.fit(np.zeros((1, n_bands)))

                n_blocks_x = (width + block_width - 1) // block_width
                n_blocks_y = (height + block_height - 1) // block_height
                total_blocks = n_blocks_x * n_blocks_y
                
                feedback.pushInfo(f"Traitement par blocs: {n_blocks_x} x {n_blocks_y} = {total_blocks} blocs")

                classification_profile = src.profile.copy()
                classification_profile.update({
                    'count': 1,
                    'dtype': 'uint8',
                    'nodata': 255
                })
                probability_profile = src.profile.copy()
                probability_profile.update({
                    'count': 1,
                    'dtype': 'float32',
                    'nodata': -9999
                })
                block_count = 0
                with rasterio.open(output_classification, 'w', **classification_profile) as dst_class, \
                     (rasterio.open(output_probability, 'w', **probability_profile) if output_probability else None) as dst_prob:
                    
                    for block_y in range(0, height, block_height):
                        for block_x in range(0, width, block_width):
                            if feedback.isCanceled():
                                return {}
                            x_size = min(block_width, width - block_x)
                            y_size = min(block_height, height - block_y)
                            
                            window = Window(block_x, block_y, x_size, y_size)
                            block = src.read(window=window)
                            if np.all(np.isnan(block)):
                                class_block = np.full((y_size, x_size), 255, dtype=np.uint8)
                                dst_class.write(class_block, 1, window=window)
                                if dst_prob is not None:
                                    prob_block = np.full((y_size, x_size), -9999, dtype=np.float32)
                                    dst_prob.write(prob_block, 1, window=window)
                                continue

                            block_reshaped = block.reshape(n_bands, -1).T
                            block_imputed = imputer.transform(block_reshaped)

                            valid_mask = ~np.any(np.isnan(block_imputed), axis=1)
                            
                            if not np.any(valid_mask):
                                class_result = np.full(block_reshaped.shape[0], 255, dtype=np.uint8)
                                prob_result = np.full(block_reshaped.shape[0], -9999, dtype=np.float32)
                            else:
                                block_scaled = scaler.transform(block_imputed)

                                class_result = np.full(block_reshaped.shape[0], 255, dtype=np.uint8)
                                prob_result = np.full(block_reshaped.shape[0], -9999, dtype=np.float32)

                                predictions = model.predict(block_scaled[valid_mask])
                                class_result[valid_mask] = predictions.astype(np.uint8)
                                if hasattr(model, 'predict_proba'):
                                    probabilities = model.predict_proba(block_scaled[valid_mask])
                                    max_probs = np.max(probabilities, axis=1)
                                    prob_result[valid_mask] = max_probs.astype(np.float32)

                            class_block = class_result.reshape(y_size, x_size)
                            dst_class.write(class_block, 1, window=window)
                            
                            if dst_prob is not None:
                                prob_block = prob_result.reshape(y_size, x_size)
                                dst_prob.write(prob_block, 1, window=window)
                            
                            block_count += 1
                            progress = int((block_count / total_blocks) * 90)
                            feedback.setProgress(progress)
                            
                            feedback.pushInfo(f"Traitement du bloc {block_count}/{total_blocks} ({progress}%)")
            feedback.pushInfo("✓ Classification terminée")
            class_layer = QgsRasterLayer(output_classification, "Classification des Cultures")
            if class_layer.isValid():
                QgsProject.instance().addMapLayer(class_layer)
                feedback.pushInfo("✓ Carte de classification ajoutée au projet")
            
            if output_probability:
                prob_layer = QgsRasterLayer(output_probability, "Probabilités de Classification")
                if prob_layer.isValid():
                    QgsProject.instance().addMapLayer(prob_layer)
                    feedback.pushInfo("✓ Carte des probabilités ajoutée au projet")
            if generate_map and output_map:
                feedback.pushInfo("Génération de la carte thématique...")
                self._generate_thematic_map(
                    output_classification, class_names, output_map, feedback
                )
            
            feedback.pushInfo("="*60)
            feedback.pushInfo("CLASSIFICATION TERMINÉE AVEC SUCCÈS")
            feedback.pushInfo("="*60)
            
            results = {self.OUTPUT_CLASSIFICATION: output_classification}
            if output_probability:
                results[self.OUTPUT_PROBABILITY] = output_probability
            if output_map:
                results[self.OUTPUT_MAP] = output_map
            
            return results
            
        except Exception as e:
            feedback.pushInfo(f"ERREUR: {str(e)}")
            raise QgsProcessingException(str(e))

    def _generate_thematic_map(self, classification_raster, class_names, output_map, feedback):
        """Generate thematic map"""
        
        try:
            with rasterio.open(classification_raster) as src:
                classification = src.read(1)
                bounds = src.bounds
                colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
                cmap = mcolors.ListedColormap(colors)

                fig, ax = plt.subplots(figsize=(12, 10))
                
                im = ax.imshow(
                    classification,
                    cmap=cmap,
                    vmin=0,
                    vmax=len(class_names)-1,
                    extent=[bounds.left, bounds.right, bounds.bottom, bounds.top]
                )
                legend_elements = [
                    Patch(facecolor=colors[i], label=class_names[i])
                    for i in range(len(class_names))
                ]
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
                ax.set_title('Carte de Classification des Cultures', fontsize=16, fontweight='bold')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_map, dpi=300, bbox_inches='tight')
                plt.close()
                
                feedback.pushInfo(f"✓ Carte thématique sauvegardée: {output_map}")
                
        except Exception as e:
            feedback.pushInfo(f"⚠️ Erreur lors de la génération de la carte: {str(e)}")

    def name(self):
        return 'classification_mapping'

    def displayName(self):
        return self.tr('4 - Classification et Cartographie')

    def group(self):
        return self.tr('Cartographie des Cultures')

    def groupId(self):
        return 'agriculture_mapping'

    def shortHelpString(self):
        return self.tr("""
        <h3>Classification et Cartographie des Cultures</h3>
        <p>Applique un modèle ML entraîné pour classifier une image satellite 
        et générer une carte des cultures.</p>
        
        <h4>Fonctionnalités:</h4>
        <ul>
        <li><b>Classification par blocs:</b> Gestion des gros rasters</li>
        <li><b>Cartes de probabilité:</b> Confiance de la classification</li>
        <li><b>Carte thématique:</b> Visualisation professionnelle</li>
        <li><b>Intégration QGIS:</b> Ajout automatique des couches</li>
        </ul>
        
        <h4>Entrées:</h4>
        <ul>
        <li><b>Image satellite:</b> Raster multi-bandes à classifier</li>
        <li><b>Modèle ML:</b> Fichier .pkl du modèle entraîné</li>
        <li><b>Scaler:</b> Fichier .pkl du scaler utilisé pour l'entraînement</li>
        <li><b>Label Encoder:</b> Fichier .pkl du label encoder utilisé pour l'entraînement</li>
        </ul>
        
        <h4>Sorties:</h4>
        <ul>
        <li><b>Classification:</b> Raster avec les classes de cultures</li>
        <li><b>Probabilités:</b> Confiance de chaque pixel (optionnel)</li>
        <li><b>Carte thématique:</b> PNG/PDF pour publication</li>
        </ul>
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def icon(self):
        """Return algorithm icon"""
        from ..help_system import HelpSystem
        help_system = HelpSystem(os.path.dirname(os.path.dirname(__file__)))
        return help_system.get_algorithm_icon('classification_mapping')

    def helpUrl(self):
        """Return help URL"""
        from ..help_system import HelpSystem
        help_system = HelpSystem(os.path.dirname(os.path.dirname(__file__)))
        return help_system.get_help_url(self.name())

    def createInstance(self):
        return ClassificationMappingAlgorithm()