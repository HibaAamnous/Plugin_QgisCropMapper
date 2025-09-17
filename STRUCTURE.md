# Structure du Plugin QGIS - Cartographie des Cultures

## Arborescence Complète

```
agriculture_mapping_plugin/          # 🌱 Dossier principal du plugin
│
├── 📄 __init__.py                   # Point d'entrée du plugin QGIS
├── 📄 agriculture_mapping_plugin.py # Classe principale du plugin
├── 📄 agriculture_processing_provider.py # Fournisseur d'algorithmes
├── 📄 metadata.txt                 # Métadonnées du plugin (version, description)
├── 📄 README.md                    # Documentation utilisateur complète
├── 📄 STRUCTURE.md                 # Structure du projet (ce fichier)
├── 📄 .gitignore                   # Fichiers à ignorer par Git
│
├── 📁 algorithms/                   # 🔬 Algorithmes de traitement
│   ├── 📄 __init__.py              # Module d'algorithmes
│   ├── 📄 satellite_acquisition.py # Acquisition d'images satellites (GEE)
│   ├── 📄 data_preparation.py      # Préparation échantillons ML
│   ├── 📄 model_training.py        # Entraînement modèles ML (4 algos)
│   ├── 📄 classification_mapping.py # Classification et cartographie
│   └── 📄 topological_correction.py # Correction topologique (Processing)
│
├── 📁 utils/                       # 🛠️ Utilitaires partagés
│   ├── 📄 __init__.py              # Module d'utilitaires
│   ├── 📄 gee_utils.py             # Utilities Google Earth Engine
│   ├── 📄 ml_utils.py              # Utilities Machine Learning
│   └── 📄 validation_utils.py      # Validation des données d'entrée
│
│
├── 📁 resources/                   # 🎨 Ressources du plugin
│   ├── 📄 resources.qrc            # Fichier de ressources Qt
│   └── 📁 icons/                   # Icônes personnalisées SVG
│       ├── 🛰️ satellite.svg        # Icône acquisition satellite
│       ├── 📊 data_prep.svg        # Icône préparation données
│       ├── 🤖 training.svg         # Icône entraînement ML
│       ├── 🗺️ classification.svg   # Icône classification
│       ├── 🔧 topology.svg         # Icône correction topologique
│       └── 📚 help.svg             # Icône aide et documentation
│
├── 📁 help/                        # 📚 Documentation complète
    ├── 📄 index.html               # Page d'aide principale
    ├── 📄 Acquisition_Satellites_help.html # Aide acquisition satellite
    ├── 📄 data_preparation_help.html # Aide préparation données
    ├── 📄 model_training_help.html  # Aide entraînement ML
    ├── 📄 classification_help.html  # Aide classification
    ├── 📄 topology_help.html       # Aide correction topologique
    └── 📄 styles.css               # Styles CSS personnalisés

└── 📄 help_system.py            # 📚 Gestionnaire système d'aide


## Détail des Composants

### 🏗️ Architecture Principale

- **`__init__.py`** : Point d'entrée requis par QGIS pour reconnaître le plugin
- **`agriculture_mapping_plugin.py`** : Classe principale gérant l'intégration QGIS
- **`agriculture_processing_provider.py`** : Gestionnaire des 5 algorithmes de traitement
- **`metadata.txt`** : Configuration du plugin (nom, version, description, auteur)

### 🔬 Algorithmes de Traitement

1. **`satellite_acquisition.py`** : Téléchargement automatique via Google Earth Engine
2. **`data_preparation.py`** : Génération d'échantillons d'entraînement géoréférencés
3. **`model_training.py`** : Comparaison de 4 algorithmes ML avec optimisation
4. **`classification_mapping.py`** : Application du meilleur modèle pour cartographier
5. **`topological_correction.py`** : Interface complète de correction des erreurs

### 🛠️ Modules Utilitaires

- **`gee_utils.py`** : Authentification, collections, téléchargement GEE
- **`ml_utils.py`** : Feature engineering, métriques, validation croisée
- **`validation_utils.py`** : Validation des entrées, gestion d'erreurs

### 🎨 Ressources Visuelles

- **Icônes SVG** : 6 icônes personnalisées pour chaque composant
- **`resources.qrc`** : Fichier de ressources Qt pour l'intégration

### 📚 Documentation

- **`index.html`** : Guide complet avec CSS personnalisé
- **`Acquisition_Satellites_help.html`** : Aide pour l'acquisition de satellites
- **`data_preparation_help.html`** : Aide pour la préparation des données
- **`model_training_help.html`** : Aide pour l'entraînement des modèles ML
- **`classification_help.html`** : Aide pour la classification et la cartographie
- **`topology_help.html`** : Aide pour la correction topologique
- **`styles.css`** : Design professionnel responsive
   
```

## 🚀 Technologies Intégrées

- **QGIS Processing Framework** : Intégration native
- **Google Earth Engine** : Données satellites en temps réel
- **Scikit-learn + XGBoost** : Machine Learning avancé
- **PyQt5** : Interface utilisateur moderne
- **GeoPandas + Shapely** : Traitement vectoriel
- **Rasterio + GDAL** : Traitement raster
- **Threading** : Opérations multi-thread

## 📦 Installation

Le plugin est entièrement autonome et s'installe comme un dossier unique dans le répertoire des plugins QGIS.