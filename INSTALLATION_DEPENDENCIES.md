Installation des Dépendances - Plugin QGIS Cartographie des Cultures
Problème Résolu : Module 'ee' non trouvé
Si vous rencontrez l'erreur ModuleNotFoundError: No module named 'ee', voici comment installer les dépendances nécessaires.

Solution Recommandée
1. Identifier l'environnement Python de QGIS
Dans QGIS, ouvrez la console Python (Extensions > Console Python) et tapez :

import sys
print(sys.executable)
print(sys.path)
Cela vous donnera le chemin vers l'exécutable Python utilisé par QGIS.

2. Installer Google Earth Engine
Option A : Via pip dans la console QGIS (Recommandé)
Dans la console Python de QGIS :

import subprocess
import sys
# Installer earthengine-api
subprocess.check_call([sys.executable, "-m", "pip", "install", "earthengine-api"])
# Redémarrer QGIS après installation

Option B : Via l'invite de commande Windows
Ouvrez l'invite de commande en tant qu'administrateur
Naviguez vers le dossier Python de QGIS :
cd "C:\Program Files\QGIS 3.40.5\apps\Python312"
Installez le package :
python.exe -m pip install earthengine-api

Option C : Via OSGeo4W Shell (Windows)
Ouvrez "OSGeo4W Shell" depuis le menu Démarrer
Tapez :
python -m pip install earthengine-api

3. Vérifier l'installation
Dans la console Python de QGIS :

try:
    import ee
    print("✓ Google Earth Engine installé avec succès")
    print(f"Version EE: {ee.__version__}")
except ImportError as e:
    print(f"❌ Erreur: {e}")

Dépendances Supplémentaires
Le plugin utilise aussi ces packages (généralement déjà installés avec QGIS) :

pip install scikit-learn  
pip install matplotlib
pip install earthengine-api 
pip install matplotlib
pip install joblib
pip install shapely
pip install seaborn  
pip install rasterio  
pip install geopandas 
pip install xgboost


Authentification Google Earth Engine
Après installation, la première utilisation nécessite une authentification :

Le plugin ouvrira automatiquement une page web pour l'authentification
Connectez-vous avec votre compte Google
Autorisez l'accès à Google Earth Engine
Copiez le code d'autorisation dans QGIS
Gestion Alternative des Erreurs
Le plugin a été modifié pour gérer l'absence de Google Earth Engine :

Si EE n'est pas installé : Message d'erreur clair avec instructions d'installation
Si EE est installé : Fonctionnement normal avec authentification automatique
Algorithmes sans EE : Les algorithmes ML et correction topologique fonctionnent indépendamment
Structure Modulaire
🌱 Plugin Cartographie des Cultures
├── 🛰️ Acquisition Satellite (nécessite EE)
├── 📊 Préparation Données (nécessite EE)  
├── 🤖 Entraînement ML (indépendant)
├── 🗺️ Classification (indépendant)
└── 🔧 Correction Topologique (indépendant)
Résolution de Problèmes Courants
Erreur : "No module named 'ee'"
Solution : Installer earthengine-api comme indiqué ci-dessus

Erreur : "Permission denied"
Solution : Lancer l'invite de commande en tant qu'administrateur

Erreur : "pip not found"
Solution : Utiliser python.exe -m pip au lieu de pip directement

Erreur d'authentification GEE
Solution : Suivre le processus d'authentification dans le navigateur

Support
Si les problèmes persistent :

Vérifiez que QGIS utilise bien Python 3.7+
Redémarrez QGIS après installation des packages
Consultez la documentation dans help/index.html
Vérifiez les logs dans la console Python de QGIS
Note importante : Le plugin fonctionne partiellement même sans Google Earth Engine. Seuls les algorithmes d'acquisition satellite et de préparation des données nécessitent EE. Les autres fonctionnalités (ML, classification, correction topologique) restent opérationnelles.