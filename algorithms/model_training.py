# -*- coding: utf-8 -*-
"""
Algorithme d'entraînement des modèles de Machine Learning
"""

import os
import matplotlib
matplotlib.use('Agg') 
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
import xgboost as xgb

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterVectorLayer, QgsProcessingParameterField,
    QgsProcessingParameterEnum, QgsProcessingParameterBoolean,
    QgsProcessingParameterFileDestination, QgsProcessingParameterFolderDestination,
    QgsProcessingException
)

from ..utils.ml_utils import prepare_training_data, evaluate_model, save_model_report
from ..utils.ml_utils import validate_training_data  

class ModelTrainingAlgorithm(QgsProcessingAlgorithm):
    """Algorithme d'entraînement des modèles ML"""
    
    INPUT_SAMPLES = 'INPUT_SAMPLES'
    CULTURE_FIELD = 'CULTURE_FIELD'
    ALGORITHMS = 'ALGORITHMS'
    OPTIMIZE_HYPERPARAMS = 'OPTIMIZE_HYPERPARAMS'
    OUTPUT_MODEL = 'OUTPUT_MODEL'
    OUTPUT_REPORT_DIR = 'OUTPUT_REPORT_DIR'

    def initAlgorithm(self, config=None):
        """Initialize algorithm parameters"""
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_SAMPLES,
                self.tr('Échantillons d\'entraînement'),
                [QgsProcessing.TypeVectorPoint]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.CULTURE_FIELD,
                self.tr('Champ des types de cultures'),
                parentLayerParameterName=self.INPUT_SAMPLES,
                type=QgsProcessingParameterField.String
            )
        )
        algorithm_options = [
            'Random Forest',
            'SVM',
            'XGBoost',
            'Decision Tree',
            'Tous les algorithmes'
        ]
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ALGORITHMS,
                self.tr('Algorithmes de Machine Learning'),
                options=algorithm_options,
                allowMultiple=True,
                defaultValue=[0, 2, 4]  
            )
        )
        
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.OPTIMIZE_HYPERPARAMS,
                self.tr('Optimiser les hyperparamètres (plus lent mais meilleur)'),
                defaultValue=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_MODEL,
                self.tr('Meilleur modèle entraîné'),
                fileFilter='Pickle files (*.pkl)'
            )
        )

        self.addParameter(
            QgsProcessingParameterFileDestination(
                'OUTPUT_SCALER',
                self.tr('Scaler entraîné'),
                fileFilter='Pickle files (*.pkl)'
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                'OUTPUT_LABEL_ENCODER',
                self.tr('Label Encoder entraîné'),
                fileFilter='Pickle files (*.pkl)'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_REPORT_DIR,
                self.tr('Dossier des rapports d\'évaluation')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Process algorithm"""
        samples_layer = self.parameterAsVectorLayer(parameters, self.INPUT_SAMPLES, context)
        culture_field = self.parameterAsString(parameters, self.CULTURE_FIELD, context)
        algorithm_indices = self.parameterAsEnums(parameters, self.ALGORITHMS, context)
        optimize_hyperparams = self.parameterAsBool(parameters, self.OPTIMIZE_HYPERPARAMS, context)
        output_model_path = self.parameterAsFileOutput(parameters, self.OUTPUT_MODEL, context)
        output_scaler_path = self.parameterAsFileOutput(parameters, 'OUTPUT_SCALER', context)
        output_label_encoder_path = self.parameterAsFileOutput(parameters, 'OUTPUT_LABEL_ENCODER', context)
        report_dir = self.parameterAsString(parameters, self.OUTPUT_REPORT_DIR, context)
        
        feedback.pushInfo("="*60)
        feedback.pushInfo("ENTRAÎNEMENT DES MODÈLES DE MACHINE LEARNING")
        feedback.pushInfo("="*60)
        
        try:
            feedback.pushInfo("Préparation des données d'entraînement...")
            X_train, X_test, y_train, y_test, feature_names, class_names = prepare_training_data(
                samples_layer, culture_field, feedback
            )
            if len(feature_names) == 0 or X_train.shape[1] == 0:
                raise QgsProcessingException(
                    "Aucune caractéristique trouvée dans les échantillons. "
                    "Assurez-vous que votre shapefile contient les bandes spectrales comme attributs.(band_1,band_2...) "
                    "Utilisez l'outil 'Préparation des Echantillons' pour extraire les valeurs des bandes."
                )
            feedback.pushInfo("Application du preprocessing...")
            imputer = SimpleImputer(strategy='mean') 
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

            scaler = RobustScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            label_encoder = LabelEncoder()
            label_encoder.fit(y_train)
            y_train = label_encoder.transform(y_train)
            y_test = label_encoder.transform(y_test)
            class_names = label_encoder.classes_ 
            if not validate_training_data(X_train, y_train, feature_names, class_names, feedback):  
                raise QgsProcessingException("Validation des données échouée. Vérifiez les échantillons.")
            feedback.pushInfo(f"✓ Données préparées:")
            feedback.pushInfo(f"  - Échantillons d'entraînement: {len(X_train)}")
            feedback.pushInfo(f"  - Échantillons de test: {len(X_test)}")
            feedback.pushInfo(f"  - Caractéristiques: {len(feature_names)}")
            feedback.pushInfo(f"  - Classes: {len(class_names)} ({', '.join(class_names)})")
            
            algorithms_to_train = self._get_algorithms_to_train(algorithm_indices)
            feedback.pushInfo(f"Algorithmes sélectionnés: {', '.join(algorithms_to_train.keys())}")

            trained_models = {}
            model_scores = {}
            
            for name, (model_class, params) in algorithms_to_train.items():
                feedback.pushInfo(f"\n--- Entraînement: {name} ---")
                
                try:
                    if optimize_hyperparams and params:
                        feedback.pushInfo("Optimisation des hyperparamètres...")
                        model = GridSearchCV(
                            model_class(),
                            params,
                            cv=3,
                            scoring='f1_weighted',
                            n_jobs=-1,
                            verbose=1
                        )
                        model.fit(X_train, y_train)
                        feedback.pushInfo(f"✓ Meilleurs paramètres: {model.best_params_}")
                        best_model = model.best_estimator_
                    else:
                        feedback.pushInfo("Entraînement avec paramètres par défaut...")
                        best_model = model_class()
                        best_model.fit(X_train, y_train)
                    
                    y_train_pred = best_model.predict(X_train) 
                    train_score = f1_score(y_train, y_train_pred, average='weighted') 

                    y_test_pred = best_model.predict(X_test) 
                    test_score = f1_score(y_test, y_test_pred, average='weighted')

                    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1_weighted')  
                    feedback.pushInfo(f"✓ F1 Score d'entraînement: {train_score:.4f}") 
                    feedback.pushInfo(f"✓ F1 Score de test: {test_score:.4f}")
                    feedback.pushInfo(f"✓ Validation croisée (F1): {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")

                    trained_models[name] = best_model
                    model_scores[name] = {
                        'train_score': train_score,
                        'test_score': test_score,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }

                    model_path = os.path.join(report_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
                    joblib.dump(trained_models[name], model_path)
                    feedback.pushInfo(f"✓ Modèle {name} sauvegardé: {model_path}")
                    
                except Exception as e:
                    feedback.pushInfo(f"❌ Erreur avec {name}: {str(e)}")
                    continue
            
            if not trained_models:
                raise QgsProcessingException("Aucun modèle n'a pu être entraîné avec succès")
            

            best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['test_score'])
            best_model = trained_models[best_model_name]
            best_score = model_scores[best_model_name]['test_score']

            feedback.pushInfo(f"\n🏆 Meilleur modèle: {best_model_name} (Score: {best_score:.4f})")
            model_data = {
                'model': best_model,
                'feature_names': feature_names,
                'class_names': class_names 
            }
            joblib.dump(model_data, output_model_path)  
            joblib.dump(scaler, output_scaler_path)  
            joblib.dump(label_encoder, output_label_encoder_path) 

            feedback.pushInfo(f"✓ Meilleur modèle sauvegardé: {output_model_path}")
            feedback.pushInfo(f"✓ Scaler sauvegardé: {output_scaler_path}")
            feedback.pushInfo(f"✓ Label Encoder sauvegardé: {output_label_encoder_path}")

            feedback.pushInfo("\nGénération des rapports d'évaluation...")
            self._generate_evaluation_reports(
                trained_models, model_scores, X_test, y_test, 
                class_names, report_dir, feedback
            )
            
            feedback.pushInfo("="*60)
            feedback.pushInfo("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
            feedback.pushInfo("="*60)
            
            return {
                self.OUTPUT_MODEL: output_model_path,
                self.OUTPUT_REPORT_DIR: report_dir
            }
            
        except Exception as e:
            feedback.pushInfo(f"ERREUR: {str(e)}")
            raise QgsProcessingException(str(e))
        

    def _get_algorithms_to_train(self, algorithm_indices):
        """Get algorithms to train based on selection"""
        
        all_algorithms = {
            'Random Forest': (
                RandomForestClassifier,
                {
                    'n_estimators': [10, 25, 50, 100],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [5, 10, 15, 20, 25, None],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'random_state': [1]
                }
            ),
            'SVM': (  
                SVC,
                {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [1, 0.1, 0.01, 0.001],
                    'kernel': ['rbf', 'poly'],
                    'probability': [True]
                }
            ),
            'XGBoost': (
                xgb.XGBClassifier,
                {
                    'min_child_weight': [1, 5, 10],
                    'gamma': [0.5, 1, 1.5, 2, 5],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'max_depth': [3, 4, 5],
                    'random_state': [1]
                }
            ),
            'Decision Tree': (
                DecisionTreeClassifier,
                {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5],
                    'random_state': [1]
                }
            )
        }
        
        selected_algorithms = {}
        
        for idx in algorithm_indices:
            if idx == 4:  
                return all_algorithms
            else:
                algo_name = list(all_algorithms.keys())[idx]
                selected_algorithms[algo_name] = all_algorithms[algo_name]
        
        return selected_algorithms

    def _generate_evaluation_reports(self, models, scores, X_test, y_test, class_names, report_dir, feedback):
        """Generate detailed evaluation reports with visualizations"""
        try:
            os.makedirs(report_dir, exist_ok=True)

            comparison_report = os.path.join(report_dir, 'model_comparison.txt')
            with open(comparison_report, 'w', encoding='utf-8') as f:
                f.write("RAPPORT DE COMPARAISON DES MODÈLES\n")
                f.write("="*50 + "\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Nombre d'échantillons de test: {len(X_test)}\n")
                f.write(f"Nombre de classes: {len(class_names)}\n")
                f.write(f"Classes: {', '.join(class_names)}\n\n")

                f.write("RÉSULTATS PAR MODÈLE:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Modèle':<20} {'Train':<10} {'Test':<10} {'CV Mean':<10} {'CV Std':<10}\n")
                f.write("-" * 80 + "\n")
                
                for name, score_data in scores.items():
                    f.write(f"{name:<20} {score_data['train_score']:<10.4f} "
                        f"{score_data['test_score']:<10.4f} {score_data['cv_mean']:<10.4f} "
                        f"{score_data['cv_std']:<10.4f}\n")
                
                f.write("-" * 80 + "\n\n")
                
                best_model_name = max(scores.keys(), key=lambda k: scores[k]['test_score'])
                f.write(f"MEILLEUR MODÈLE: {best_model_name}\n")
                f.write(f"Score de test: {scores[best_model_name]['test_score']:.4f}\n\n")

            for name, model in models.items():
                feedback.pushInfo(f"Génération du rapport pour {name}...")
                
                y_pred = model.predict(X_test)

                evaluation_results = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred, target_names=class_names, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'predictions': y_pred,
                    'true_labels': y_test
                }
                report_path, cm_image_path = save_model_report(
                    model, name, evaluation_results, 
                    self.feature_names, class_names, report_dir 
                )
                
                feedback.pushInfo(f"✓ Rapport généré: {os.path.basename(report_path)}")
                feedback.pushInfo(f"✓ Matrice de confusion: {os.path.basename(cm_image_path)}")
                    
            feedback.pushInfo(f"✓ Rapports et visualisations générés dans: {report_dir}")
            
        except Exception as e:
            feedback.pushInfo(f"❌ Erreur lors de la génération des rapports: {str(e)}")
 

    def name(self):
        return 'model_training'

    def displayName(self):
        return self.tr('3 - Entraînement Modèles ML')

    def group(self):
        return self.tr('Cartographie des Cultures')

    def groupId(self):
        return 'agriculture_mapping'

    def shortHelpString(self):
        return self.tr("""
        <h3>Entraînement des Modèles de Machine Learning</h3>
        <p>Entraîne et évalue plusieurs algorithmes de classification pour la cartographie des cultures.</p>
        
        <h4>Algorithmes disponibles:</h4>
        <ul>
        <li><b>Random Forest:</b> Ensemble de arbres de décision</li>
        <li><b>SVM:</b> Machine à vecteurs de support</li>
        <li><b>XGBoost:</b> Gradient boosting optimisé</li>
        <li><b>Decision Tree:</b> Arbre de décision simple</li>
        </ul>
        
        <h4>Fonctionnalités:</h4>
        <ul>
        <li><b>Optimisation automatique:</b> GridSearch pour les hyperparamètres</li>
        <li><b>Validation croisée:</b> Évaluation robuste des performances</li>
        <li><b>Sélection automatique:</b> Meilleur modèle basé sur le f1 score</li>
        <li><b>Rapports détaillés:</b> Métriques et matrices de confusion</li>
        </ul>
        
        <h4>Sorties:</h4>
        <ul>
        <li><b>Modèle optimal:</b> Fichier .pkl avec le meilleur algorithme</li>
        <li><b>Rapports:</b> Évaluation complète de tous les modèles</li>
        </ul>
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def icon(self):
        """Return algorithm icon"""
        from ..help_system import HelpSystem
        help_system = HelpSystem(os.path.dirname(os.path.dirname(__file__)))
        return help_system.get_algorithm_icon('model_training')

    def helpUrl(self):
        """Return help URL"""
        from ..help_system import HelpSystem
        help_system = HelpSystem(os.path.dirname(os.path.dirname(__file__)))
        return help_system.get_help_url(self.name())

    def createInstance(self):
        return ModelTrainingAlgorithm()