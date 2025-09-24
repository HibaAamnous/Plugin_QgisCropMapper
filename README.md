# 🌱 Plugin QGIS de Cartographie des Cultures

## Machine Learning & Google Earth Engine

Un plugin QGIS professionnel développé au **Pôle Digital de l'Agriculture, de la Forêt et Observatoire de la Sécheresse** pour automatiser la cartographie des cultures agricoles.

## 🎯 Fonctionnalités Principales

- **🛰️ Acquisition Satellite** : Téléchargement automatique via Google Earth Engine (Sentinel-2)
- **📊 Préparation Données** : Génération d'échantillons d'entraînement géoréférencés
- **🤖 Machine Learning** : 4 algorithmes optimisés (Random Forest, SVM, XGBoost, Decision Tree)
- **🗺️ Classification** : Application du meilleur modèle avec cartes thématiques
- **🔧 Correction Topologique** : Interface graphique complète pour corriger les erreurs géométriques

## 📁 Structure du Plugin

```
agriculture_mapping_plugin/
├── __init__.py                          # Point d'entrée du plugin
├── agriculture_mapping_plugin.py        # Classe principale du plugin
├── agriculture_processing_provider.py   # Fournisseur d'algorithmes
├── metadata.txt                         # Métadonnées du plugin
├── README.md                           # Documentation (ce fichier)
├── replit.md                           # Configuration du projet
│
├── algorithms/                         # Algorithmes de traitement
│   ├── satellite_acquisition.py       # Acquisition d'images satellites
│   ├── data_preparation.py            # Préparation des échantillons
│   ├── model_training.py              # Entraînement des modèles ML
│   ├── classification_mapping.py      # Classification et cartographie
│   └── topological_correction.py      # Correction topologique
│
├── utils/                              # Utilitaires partagés
│   ├── gee_utils.py                   # Utilities Google Earth Engine
│   ├── ml_utils.py                    # Utilities Machine Learning
│   └── validation_utils.py            # Validation des données
│
├── resources/                         # Ressources du plugin
│   ├── resources.qrc                  # Fichier de ressources Qt
│   └── icons/                         # Icônes personnalisées
│       ├── satellite.svg              # Icône acquisition satellite
│       ├── data_prep.svg              # Icône préparation données
│       ├── training.svg               # Icône entraînement ML
│       ├── classification.svg         # Icône classification
│       ├── topology.svg               # Icône correction topologique
│       └── help.svg                   # Icône aide
│
└── help/                              # Documentation complète
    ├── index.html                     # Page d'aide principale
    └── styles.css                     # Styles CSS personnalisés
```

## ⚙️ Installation

### Prérequis
- **QGIS 3.16+** (Version LTR recommandée)
- **Python 3.7+** avec bibliothèques géospatiales
- **Compte Google Earth Engine** ([S'inscrire ici](https://earthengine.google.com/))
- **Connexion Internet** pour l'accès aux données satellites

### Installation du Plugin

1. **Télécharger** le dossier complet `agriculture_mapping_plugin`

2. **Copier** dans le dossier des plugins QGIS :
   - **Windows:** `C:\Users\[utilisateur]\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
   - **Linux:** `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - **macOS:** `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`

3. **Redémarrer** QGIS

4. **Activer** le plugin dans : *Extensions > Gérer/Installer les extensions*

5. **Rechercher** et activer : *"CartoCultures_ML_GEE"*

### ⚠️ Installation des Dépendances 

📖 **Guide complet** : Consultez `INSTALLATION_DEPENDENCIES.md` pour plus de détails.

**Installation rapide via OSGeo4W Shell :**
```bash
python -m pip install --upgrade pip
python -m pip install scikit-learn xgboost python-docx
python -m pip install seaborn matplotlib earthengine-api
python -m pip install rasterio joblib python-magic
```


## 🚀 Utilisation

### Pipeline Complet de Cartographie

1. **🛰️ Acquisition Satellite** → Téléchargement d'images filtrées
2. **📊 Préparation Données** → Génération d'échantillons d'entraînement  
3. **🤖 Entraînement ML** → Comparaison de 4 algorithmes
4. **🗺️ Classification** → Application du meilleur modèle
5. **🔧 Correction Topologique** → Nettoyage des erreurs géométriques

### Accès aux Algorithmes

- **Menu principal :** *Extensions > CartoCultures_ML_GEE*
- **Boîte à outils :** *Traitement > CartoCultures_ML_GEE*

## 🔧 Configuration Google Earth Engine

Lors de la première utilisation :
1. Une fenêtre d'authentification s'ouvrira automatiquement
2. Suivre les instructions pour autoriser l'accès
3. L'authentification est sauvegardée pour les utilisations futures

## 📚 Documentation Complète

Ouvrir le fichier `help/index.html` dans un navigateur pour accéder à :
- Guide d'installation détaillé
- Tutoriels étape par étape
- Description des algorithmes
- Résolution de problèmes

## 🤖 Algorithmes Machine Learning

- **🌳 Random Forest** : Ensemble d'arbres robuste et interprétable
- **⚡ SVM** : Machine à vecteurs de support efficace et précise
- **🚀 XGBoost** : Gradient boosting optimisé et performant
- **🌲 Decision Tree** : Arbre de décision simple et rapide

Tous avec optimisation automatique des hyperparamètres et validation croisée.

## 🛰️ Satellites Supportés

- **Sentinel-2** : Résolution 10m, bandes B2-B12, revisit 5 jours


## 📄 Licence

Plugin développé pour le Pôle Digital de l'Agriculture, de la Forêt et Observatoire de la Sécheresse.

## 📞 Support

Pour toute question ou problème :
- **Documentation** : Consulter `help/index.html`
- **Issues** : Signaler les problèmes techniques
- **Email** : hibaamnous@gmail.com

---

**Version 2.0.2** - Plugin QGIS professionnel de cartographie des cultures
