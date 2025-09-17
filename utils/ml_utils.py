# -*- coding: utf-8 -*-
"""
Utilitaires pour le Machine Learning
"""
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non interactif pour éviter les problèmes avec QGIS
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
try:
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import os
from datetime import datetime


def check_ml_dependencies():
    """Vérifier que toutes les dépendances ML sont disponibles"""
    missing_deps = []

    if not SKLEARN_AVAILABLE:
        missing_deps.append("scikit-learn")

    if not XGBOOST_AVAILABLE:
        missing_deps.append("xgboost")

    if missing_deps:
        error_msg = f"Modules manquants : {', '.join(missing_deps)}\n"
        error_msg += "Installez avec : pip install " + " ".join(missing_deps)
        raise ImportError(error_msg)

    return True


def prepare_training_data(samples_layer, culture_field, feedback, test_size=0.3, random_state=42):
    """
    Préparer les données d'entraînement à partir d'une couche QGIS

    Args:
        samples_layer: Couche QGIS avec les échantillons
        culture_field: Nom du champ contenant les types de cultures
        feedback: Objet feedback pour les messages
        test_size: Proportion des données de test
        random_state: Graine pour la reproductibilité

    Returns:
        X_train, X_test, y_train, y_test, feature_names, class_names
    """
    try:
        features_data = []
        labels = []
        fields = samples_layer.fields()
        feature_fields = []

        for field in fields:
            field_name = field.name()
            if (field_name.startswith('band_') and  
                field_name != culture_field and 
                field_name not in ['Train', 'fid', 'id', 'geometry'] and
                field.type() in [2, 3, 4, 6]):  
                feature_fields.append(field_name)

        feedback.pushInfo(f"Caractéristiques détectées: {feature_fields}")

        for feature in samples_layer.getFeatures():
            feature_values = []
            for field_name in feature_fields:
                value = feature[field_name]
                if value is None:
                    value = 0 
                feature_values.append(float(value))
            label = feature[culture_field]
            if label is not None and label != '':
                features_data.append(feature_values)
                labels.append(str(label))

        if len(features_data) == 0:
            raise Exception("Aucune donnée d'entraînement valide trouvée")
        X = np.array(features_data)
        y = np.array(labels)
        if np.any(np.isnan(X)):
            nan_count = np.sum(np.isnan(X))
            feedback.pushInfo(f"⚠️ Valeurs manquantes détectées: {nan_count}. Imputation avec mean.")
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)  
        unique_classes = np.unique(y)
        feedback.pushInfo(f"Classes détectées: {list(unique_classes)}")

        if 'Train' in [field.name() for field in fields]:
            feedback.pushInfo("Division train/test existante détectée")

            train_mask = []
            for feature in samples_layer.getFeatures():
                train_value = feature['Train']
                train_mask.append(train_value is True or str(train_value).lower() == 'true')
            train_mask = np.array(train_mask)
            test_mask = ~train_mask

            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]

        else:
            feedback.pushInfo(f"Division automatique train/test ({int((1-test_size)*100)}%/{int(test_size*100)}%)")
            if SKLEARN_AVAILABLE:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
            else:
                raise ImportError("scikit-learn n'est pas disponible, train_test_split ne peut pas être appelé.")

        feedback.pushInfo(f"Données d'entraînement: {len(X_train)} échantillons")
        feedback.pushInfo(f"Données de test: {len(X_test)} échantillons")

        return X_train, X_test, y_train, y_test, feature_fields, list(unique_classes)

    except Exception as e:
        feedback.pushInfo(f"Erreur dans prepare_training_data: {str(e)}")
        raise


def evaluate_model(model, X_test, y_test, class_names):
    """
    Évaluer un modèle de ML

    Args:
        model: Modèle entraîné
        X_test: Données de test
        y_test: Labels de test
        class_names: Noms des classes

    Returns:
        dict: Dictionnaire avec les métriques d'évaluation
    """
    try:
        y_pred = model.predict(X_test)
        if SKLEARN_AVAILABLE:
            accuracy = accuracy_score(y_test, y_pred)
        else:
            raise ImportError("scikit-learn n'est pas disponible, accuracy_score ne peut pas être appelé.")

        if SKLEARN_AVAILABLE:
            class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        else:
             raise ImportError("scikit-learn n'est pas disponible, classification_report ne peut pas être appelé.")

        if SKLEARN_AVAILABLE:
            conf_matrix = confusion_matrix(y_test, y_pred)
        else:
             raise ImportError("scikit-learn n'est pas disponible, confusion_matrix ne peut pas être appelé.")

        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)

        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'true_labels': y_test,
            'probabilities': probabilities
        }

    except Exception as e:
        raise Exception(f"Erreur dans evaluate_model: {str(e)}")


def save_model_report(model, model_name, evaluation_results, feature_names, class_names, output_dir):
    """
    Sauvegarder un rapport détaillé du modèle

    Args:
        model: Modèle entraîné
        model_name: Nom du modèle
        evaluation_results: Résultats de l'évaluation
        feature_names: Noms des caractéristiques
        class_names: Noms des classes
        output_dir: Dossier de sortie
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{model_name.lower().replace(' ', '_')}_report_{timestamp}.txt"
        report_path = os.path.join(output_dir, report_filename)
        
        cm_image_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_cm_{timestamp}.png")
        plot_confusion_matrix(
            evaluation_results['true_labels'], 
            evaluation_results['predictions'], 
            class_names, 
            cm_image_path,
            title=f"Matrice de Confusion - {model_name}"
        )

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"RAPPORT D'ÉVALUATION - {model_name}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Modèle: {model_name}\n")
            f.write(f"Précision globale: {evaluation_results['accuracy']:.4f}\n\n")
            f.write("CARACTÉRISTIQUES UTILISÉES:\n")
            f.write("-" * 30 + "\n")
            for i, feature in enumerate(feature_names, 1):
                f.write(f"{i:2d}. {feature}\n")
            f.write("\n")
            f.write("CLASSES:\n")
            f.write("-" * 10 + "\n")
            for i, class_name in enumerate(class_names, 1):
                f.write(f"{i:2d}. {class_name}\n")
            f.write("\n")
            f.write("RAPPORT DE CLASSIFICATION DÉTAILLÉ:\n")
            f.write("-" * 40 + "\n")
            class_report = evaluation_results['classification_report']
            f.write(f"{'Classe':<15} {'Précision':<10} {'Rappel':<10} {'F1-Score':<10} {'Support':<10}\n")
            f.write("-" * 60 + "\n")
            for class_name in class_names:
                if class_name in class_report:
                    metrics = class_report[class_name]
                    f.write(f"{class_name:<15} {metrics['precision']:<10.3f} "
                           f"{metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} "
                           f"{metrics['support']:<10.0f}\n")

            f.write("-" * 60 + "\n")
            if 'macro avg' in class_report:
                macro = class_report['macro avg']
                f.write(f"{'Macro avg':<15} {macro['precision']:<10.3f} "
                       f"{macro['recall']:<10.3f} {macro['f1-score']:<10.3f} "
                       f"{macro['support']:<10.0f}\n")

            if 'weighted avg' in class_report:
                weighted = class_report['weighted avg']
                f.write(f"{'Weighted avg':<15} {weighted['precision']:<10.3f} "
                       f"{weighted['recall']:<10.3f} {weighted['f1-score']:<10.3f} "
                       f"{weighted['support']:<10.0f}\n")
            f.write("\n")

            f.write("MATRICE DE CONFUSION:\n")
            f.write("-" * 25 + "\n")
            conf_matrix = evaluation_results['confusion_matrix']

            f.write(f"{'Réel / Prédit':<15}")
            for class_name in class_names:
                f.write(f"{class_name:<10}")
            f.write("\n")

            for i, class_name in enumerate(class_names):
                f.write(f"{class_name:<15}")
                for j in range(len(class_names)):
                    f.write(f"{conf_matrix[i,j]:<10}")
                f.write("\n")
            f.write("\n")

            if hasattr(model, 'feature_importances_'):
                f.write("IMPORTANCE DES CARACTÉRISTIQUES:\n")
                f.write("-" * 35 + "\n")

                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]

                for i in range(len(feature_names)):
                    idx = indices[i]
                    f.write(f"{i+1:2d}. {feature_names[idx]:<20} {importances[idx]:.4f}\n")
                f.write("\n")

            f.write("PARAMÈTRES DU MODÈLE:\n")
            f.write("-" * 20 + "\n")
            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param, value in params.items():
                    f.write(f"{param}: {value}\n")

        return report_path, cm_image_path

    except Exception as e:
        raise Exception(f"Erreur dans save_model_report: {str(e)}")


def validate_training_data(X, y, feature_names, class_names, feedback):
    """
    Valider les données d'entraînement

    Args:
        X: Matrice des caractéristiques
        y: Vecteur des labels
        feature_names: Noms des caractéristiques
        class_names: Noms des classes
        feedback: Objet feedback pour les messages

    Returns:
        bool: True si les données sont valides
    """
    try:
        feedback.pushInfo("Validation des données d'entraînement...")
        if len(X) == 0:
            feedback.pushInfo("❌ Erreur: Aucune donnée d'entraînement")
            return False

        if len(X) != len(y):
            feedback.pushInfo("❌ Erreur: Nombre d'échantillons incohérent")
            return False

        if X.shape[1] != len(feature_names):
            feedback.pushInfo("❌ Erreur: Nombre de caractéristiques incohérent")
            return False
        if np.any(np.isnan(X)):
            nan_count = np.sum(np.isnan(X))
            feedback.pushInfo(f"⚠️ Attention: {nan_count} valeurs manquantes détectées")

        unique_labels, counts = np.unique(y, return_counts=True)
        feedback.pushInfo("Distribution des classes:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(y)) * 100
            feedback.pushInfo(f"  {label}: {count} échantillons ({percentage:.1f}%)")

        min_samples = np.min(counts)
        max_samples = np.max(counts)
        imbalance_ratio = max_samples / min_samples

        if imbalance_ratio > 10:
            feedback.pushInfo(f"⚠️ Attention: Classes très déséquilibrées (ratio: {imbalance_ratio:.1f})")
        elif imbalance_ratio > 3:
            feedback.pushInfo(f"⚠️ Attention: Classes déséquilibrées (ratio: {imbalance_ratio:.1f})")
        else:
            feedback.pushInfo("✓ Classes relativement équilibrées")

        feedback.pushInfo("Analyse des caractéristiques:")
        for i, feature_name in enumerate(feature_names):
            feature_values = X[:, i]
            if np.all(feature_values == feature_values[0]):
                feedback.pushInfo(f"⚠️ Caractéristique constante: {feature_name}")
            elif np.std(feature_values) < 1e-6:
                feedback.pushInfo(f"⚠️ Caractéristique peu variable: {feature_name}")

        feedback.pushInfo("✓ Validation des données terminée")
        return True

    except Exception as e:
        feedback.pushInfo(f"❌ Erreur lors de la validation: {str(e)}")
        return False

def plot_confusion_matrix(y_true, y_pred, class_names, output_path, normalize='true', title='Matrice de Confusion'):
    """
    Générer et sauvegarder une visualisation de la matrice de confusion
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions du modèle
        class_names: Noms des classes
        output_path: Chemin de sauvegarde de l'image
        normalize: Méthode de normalisation ('true', 'pred', None)
        title: Titre du graphique
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', ax=ax, values_format='.2f' if normalize else 'd')
        ax.set_title(title, fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    except Exception as e:
        raise Exception(f"Erreur lors de la création de la matrice de confusion: {str(e)}")