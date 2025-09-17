# -*- coding: utf-8 -*-
"""
Utilitaires pour Google Earth Engine - Version Sentinel-2 
"""

try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    ee = None
import uuid
def ensure_gee_initialized(feedback=None, project_id=None):
    """
    Initialiser Google Earth Engine avec gestion d'erreurs robuste
    """
    if not EE_AVAILABLE:
        error_msg = ("Google Earth Engine n'est pas installé. "
                    "Installez le package 'earthengine-api' avec: pip install earthengine-api")
        if feedback:
            feedback.pushInfo(f"❌ {error_msg}")
        raise Exception(error_msg)
    
    try:
        if project_id:
            ee.Initialize(project=project_id)
            if feedback:
                feedback.pushInfo("✓ Google Earth Engine initialisé avec succès")
            return
        else:
            raise Exception("❌ ID de projet Google Cloud requis. Créez-en un ici : https://console.cloud.google.com/projectcreate")
            
    except Exception as first_error:
        if feedback:
            feedback.pushInfo("🔐 Authentification Google Earth Engine requise...")
        
        try:
            ee.Authenticate(force=True)
            if feedback:
                feedback.pushInfo("✓ Authentification terminée")
            if not project_id:
                raise Exception("❌ Après authentification, vous DEVEZ spécifier un ID de projet")
            ee.Initialize(project=project_id)  
            if feedback:
                feedback.pushInfo("✓ Google Earth Engine initialisé avec succès")
                
        except Exception as error:
            error_msg = (
                "Échec de l'initialisation GEE:\n"
                "1. Créez un projet sur Google Cloud Console\n"
                "2. Activez l'API Earth Engine\n"
                "3. Spécifiez votre ID de projet (ex: 'mon-projet-123')\n"
                f"Erreur technique: {str(error)}"
            )
            if feedback:
                feedback.pushInfo(f"❌ {error_msg}")
            raise Exception(error_msg)


def get_available_indices():
    """
    Obtenir la liste des indices de végétation disponibles pour la cartographie des cultures
    """
    indices = {
        'NDVI': {
            'name': 'Normalized Difference Vegetation Index',
            'description': 'Indice de végétation le plus utilisé pour la cartographie des cultures',
            'formula': '(NIR - RED) / (NIR + RED)',
            'bands': ['B8', 'B4']
        },
        'SAVI': {
            'name': 'Soil Adjusted Vegetation Index', 
            'description': 'Indice ajusté pour réduire l\'effet du sol, idéal pour cultures éparses',
            'formula': '(1 + L) * (NIR - RED) / (NIR + RED + L), L=0.5',
            'bands': ['B8', 'B4']
        },
        'EVI': {
            'name': 'Enhanced Vegetation Index',
            'description': 'Indice amélioré, moins sensible aux effets atmosphériques',
            'formula': '2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)',
            'bands': ['B8', 'B4', 'B2']
        },
        'NDWI': {
            'name': 'Normalized Difference Water Index',
            'description': 'Indice pour détecter l\'eau et l\'humidité des cultures',
            'formula': '(GREEN - NIR) / (GREEN + NIR)',
            'bands': ['B3', 'B8']
        },
        'GNDVI': {
            'name': 'Green Normalized Difference Vegetation Index',
            'description': 'Variante du NDVI utilisant le vert, sensible à la chlorophylle',
            'formula': '(NIR - GREEN) / (NIR + GREEN)',
            'bands': ['B8', 'B3']
        },
        'MNDWI': {
            'name': 'Modified Normalized Difference Water Index',
            'description': 'Version modifiée pour mieux détecter les surfaces en eau',
            'formula': '(GREEN - SWIR1) / (GREEN + SWIR1)',
            'bands': ['B3', 'B11']
        }
    }
    
    return indices

def get_sentinel2_metadata():
    """
    Obtenir les métadonnées de la collection Sentinel-2
    """
    return {
        'id': 'COPERNICUS/S2_SR_HARMONIZED',
        'name': 'Sentinel-2 Surface Reflectance Harmonized',
        'resolution': 10,
        'cloud_property': 'CLOUDY_PIXEL_PERCENTAGE',
        'bands': {
            'blue': 'B2',
            'green': 'B3', 
            'red': 'B4',
            'nir': 'B8',
            'swir1': 'B11',
            'swir2': 'B12'
        },
        'scale_factor': 0.0001
    }
def calculate_cloud_and_nodata_metrics(image, geometry):
    """
    Calcul de pourcentage de cloud et no data dans la zone d'etude 
    """
    def get_fallback_metrics(img):
        """Fonction helper pour obtenir les métriques de fallback"""
        native_cloud_cover = img.get('CLOUDY_PIXEL_PERCENTAGE')
        native_cloud_cover = ee.Number(ee.Algorithms.If(
            ee.Algorithms.IsEqual(native_cloud_cover, None), 
            100, 
            native_cloud_cover
        ))
        
        return img.set({
            'CLOUDY_PIXEL_PERCENTAGE': native_cloud_cover,
            'current_nodata_percentage': 0,
            'current_cloud_percentage': native_cloud_cover
        })
    
    try:
        band_names = image.bandNames()
        has_scl = band_names.contains('SCL')
        has_b8 = band_names.contains('B8')
        
        condition = ee.Algorithms.And(has_scl, has_b8)

        if condition.getInfo():
            masked_image = image.select('B8').mask()
            scl = image.select('SCL')
            cloud_mask = ee.Algorithms.Or(
                ee.Algorithms.Or(scl.eq(8), scl.eq(9)),
                scl.eq(10)
            )
            
            total_count = masked_image.gt(0).reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=geometry,
                scale=10,
                maxPixels=1e13,
                bestEffort=True
            ).get('B8')

            valid_count = image.select('B8').reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=geometry,
                scale=10,
                maxPixels=1e13,
                bestEffort=True
            ).get('B8')

            cloud_count = cloud_mask.updateMask(cloud_mask).reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=geometry,
                scale=10,
                maxPixels=1e13,
                bestEffort=True
            ).get('SCL')
                    
            total_count = ee.Number(total_count)
            valid_count = ee.Number(valid_count)
            cloud_count = ee.Number(cloud_count)
            
            nodata_count = total_count.subtract(valid_count)
            nodata_percent = ee.Algorithms.If(
                total_count.gt(0),
                nodata_count.multiply(100).divide(total_count),
                100  # Si total_count est 0, considérer 100% de données manquantes
            )
            
            cloud_percent = ee.Algorithms.If(
                valid_count.gt(0),
                cloud_count.multiply(100).divide(valid_count),
                100  # Si valid_count est 0, considérer 100% de nuages
            )
            
            native_cloud_cover = image.get('CLOUDY_PIXEL_PERCENTAGE')
            native_cloud_cover = ee.Number(ee.Algorithms.If(
                ee.Algorithms.IsEqual(native_cloud_cover, None), 
                100, 
                native_cloud_cover
            ))
            
            clipped = image.clip(geometry)

            return clipped.set({
                'CLOUDY_PIXEL_PERCENTAGE': native_cloud_cover,
                'current_nodata_percentage': nodata_percent,
                'current_cloud_percentage': cloud_percent
            }).addBands(cloud_mask.rename('MASKCLOUD'))
        else:
            return get_fallback_metrics(image)
            
    except Exception as e:
        error_msg = f"Erreur dans calculate_cloud_and_nodata_metrics: {str(e)}"

        return get_fallback_metrics(image)
    
def debug_image_properties(collection, feedback=None):
    """Debug des propriétés des images"""
    if feedback:
        feedback.pushInfo("=== DEBUG PROPRIÉTÉS ===")
    
    image_list = collection.toList(collection.size())
    n_images = image_list.size().getInfo()
    
    for i in range(min(3, n_images)):  
        image = ee.Image(image_list.get(i))
        props = image.toDictionary().getInfo()
        
        if feedback:
            feedback.pushInfo(f"Image {i+1} propriétés:")
            for key, value in props.items():
                if 'cloud' in key.lower() or 'nodata' in key.lower():
                    feedback.pushInfo(f"  {key}: {value}")

def calculate_vegetation_indices_s2(image, geometry, indices_list=None, feedback=None):
    """
    Calculer les indices de végétation pour Sentinel-2 
    """
    if indices_list is None:
        indices_list = ['NDVI']
    
    try:
        if feedback:
            feedback.pushInfo(f"Calcul des indices: {', '.join(indices_list)}")
        
        result_image = image
        
        for index in indices_list:
            if index == 'NDVI':
                ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi')
                result_image = result_image.addBands(ndvi)
                if feedback:
                    feedback.pushInfo("✓ NDVI calculé")
            
            elif index == 'SAVI':
                L = 0.5  # Facteur d'ajustement du sol
                savi = image.expression(
                    '(1 + L) * (NIR - RED) / (NIR + RED + L)',
                    {
                        'NIR': image.select('B8'),
                        'RED': image.select('B4'),
                        'L': L
                    }
                ).rename('savi')
                result_image = result_image.addBands(savi)
                if feedback:
                    feedback.pushInfo("✓ SAVI calculé")
            
            elif index == 'EVI':
                evi = image.expression(
                    '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
                    {
                        'NIR': image.select('B8'),
                        'RED': image.select('B4'),
                        'BLUE': image.select('B2')
                    }
                ).rename('evi')
                result_image = result_image.addBands(evi)
                if feedback:
                    feedback.pushInfo("✓ EVI calculé")
            
            elif index == 'NDWI':
                ndwi = image.normalizedDifference(['B3', 'B8']).rename('ndwi')
                result_image = result_image.addBands(ndwi)
                if feedback:
                    feedback.pushInfo("✓ NDWI calculé")
            
            elif index == 'GNDVI':
                gndvi = image.normalizedDifference(['B8', 'B3']).rename('gndvi')
                result_image = result_image.addBands(gndvi)
                if feedback:
                    feedback.pushInfo("✓ GNDVI calculé")
            
            elif index == 'MNDWI':
                mndwi = image.normalizedDifference(['B3', 'B11']).rename('mndwi')
                result_image = result_image.addBands(mndwi)
                if feedback:
                    feedback.pushInfo("✓ MNDWI calculé")
        
        return result_image
        
    except Exception as e:
        error_msg = f"Erreur lors du calcul des indices: {str(e)}"
        if feedback:
            feedback.pushInfo(f"❌ {error_msg}")
        raise Exception(error_msg)

def create_water_and_built_mask(collection, geometry, feedback=None):
    """
    Créer les masques d'eau et de bâti 
    """
    try:
        if feedback:
            feedback.pushInfo("Création des masques eau et bâti...")
        # Masque bâti avec WorldCover v200
        world_cover = ee.ImageCollection('ESA/WorldCover/v200') \
                       .sort('system:time_start', False) \
                       .first() \
                       .select('Map')
        
        built_mask = world_cover.eq(50).unmask(0)
        median = collection.median()
        mndwi = median.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        water_mask = mndwi.gt(0.3).unmask(0)
        full_mask = built_mask.Or(water_mask).rename('MASK_TOTAL')
        
        if feedback:
            feedback.pushInfo("✓ Masques créés avec succès")
        
        return water_mask, built_mask, full_mask
        
    except Exception as e:
        error_msg = f"Erreur lors de la création des masques: {str(e)}"
        if feedback:
            feedback.pushInfo(f"❌ {error_msg}")
        raise Exception(error_msg)

def process_indices_time_series(collection, geometry, indices_list=['NDVI'], apply_mask=False, feedback=None):
    """
    Traiter une série temporelle d'indices de végétation
    """
    try:
        if feedback:
            feedback.pushInfo(f"Traitement de la série temporelle: {', '.join(indices_list)}")
        
        def make_indices(img):
            img_with_indices = calculate_vegetation_indices_s2(img, geometry, indices_list)
            band_name = indices_list[0].lower()
            result = img_with_indices.select([band_name]).toFloat().rename(band_name)
            time_start = img.get('system:time_start')
            return result.set('system:time_start', time_start)


        indices_collection = collection.map(make_indices)
        indices_collection = indices_collection.sort('system:time_start')
        indices_img = indices_collection.toBands()
        
        if not apply_mask:
            return indices_img
        water_mask, built_mask, full_mask = create_water_and_built_mask(collection, geometry, feedback)
        indices_masked = indices_img.updateMask(full_mask.Not())
        
        if feedback:
            feedback.pushInfo(f"✓ Série temporelle {', '.join(indices_list)} traitée avec succès")
        
        return indices_img, indices_masked
        
    except Exception as e:
        error_msg = f"Erreur lors du traitement des indices: {str(e)}"
        if feedback:
            feedback.pushInfo(f"❌ {error_msg}")
        raise Exception(error_msg)

def download_sentinel2_image(image, geometry, description_prefix, feedback=None):
    """
    Télécharger une image Sentinel-2 depuis GEE via Google Drive
    """
    try:
        if feedback:
            feedback.pushInfo("🔁 Lancement de l'export vers Google Drive...")
        
        clean_prefix = description_prefix
        clean_prefix = clean_prefix.replace("é", "e").replace("è", "e").replace("ê", "e")
        clean_prefix = clean_prefix.replace("à", "a").replace("â", "a")
        clean_prefix = clean_prefix.replace("î", "i").replace("ï", "i")
        clean_prefix = clean_prefix.replace("ô", "o").replace("ö", "o")
        clean_prefix = clean_prefix.replace("ù", "u").replace("û", "u").replace("ü", "u")
        clean_prefix = clean_prefix.replace("ç", "c")
        clean_prefix = clean_prefix.replace(" ", "_").replace("-", "_")
        import re
        clean_prefix = re.sub(r'[^a-zA-Z0-9_\.\,\:\;\-\_]', '', clean_prefix)
        clean_prefix = clean_prefix[:100]
        task_id = f"{clean_prefix}_{uuid.uuid4().hex[:8]}"
        bounds = geometry.bounds()
        region = bounds.getInfo()['coordinates']
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=task_id,
            fileNamePrefix=task_id,
            region=region,  
            scale=10,
            fileFormat='GeoTIFF',
            maxPixels=1e13
        )
        task.start()
        if feedback:
            feedback.pushInfo(f"✅ Tâche d'export créée: {task_id}")
            feedback.pushInfo("ℹ️ L'image est en cours d'export vers VOTRE Google Drive")
            feedback.pushInfo("📁 Récupérez le fichier ici : https://drive.google.com/drive/my-drive")
            feedback.pushInfo(f"🔎 Cherchez le fichier : {task_id}.tif")
            feedback.pushInfo("⏱️ Cette opération peut prendre plusieurs minutes")
            feedback.pushInfo(f"🗺️ Emprise d'export: {region}")
        
    except Exception as e:
        error_msg = f"Erreur lors de l'export: {str(e)}"
        if feedback:
            feedback.pushInfo(f"❌ {error_msg}")
        raise Exception(error_msg)

def get_filtered_sentinel2_collection(geometry, start_date, end_date, max_cloud_cover=20, max_nodata=5, feedback=None):
    try:
        if feedback:
            feedback.pushInfo("Chargement de la collection Sentinel-2...")
            feedback.pushInfo(f"Période: {start_date} à {end_date}")
            feedback.pushInfo(f"Géométrie type: {type(geometry)}")
        try:
            area = geometry.area(1).getInfo()
            if feedback:
                feedback.pushInfo(f"Superficie de la zone: {area:.2f} m²")
            coords_info = geometry.getInfo()
            if feedback:
                feedback.pushInfo(f"Type de géométrie: {coords_info.get('type', 'inconnu')}")
                
        except Exception as ee_error:
            if feedback:
                feedback.pushInfo(f"Erreur validation géométrie: {str(ee_error)}")
        
        s2_meta = get_sentinel2_metadata()
        collection = ee.ImageCollection(s2_meta['id']) \
                      .filterDate(start_date, end_date) \
                      .filterBounds(geometry)
        
        initial_count = collection.size().getInfo()
        if feedback:
            feedback.pushInfo(f"Images initiales: {initial_count}")
        
        if initial_count == 0:
            extended_start_date = ee.Date(start_date).advance(-15, 'day').format().getInfo()
            extended_collection = ee.ImageCollection(s2_meta['id']) \
                                  .filterDate(extended_start_date, end_date) \
                                  .filterBounds(geometry)
            extended_count = extended_collection.size().getInfo()
            
            if extended_count == 0:
                raise Exception("Aucune image Sentinel-2 trouvée pour la période et la zone spécifiées.")
            else:
                if feedback:
                    feedback.pushInfo(f"Aucune image trouvée pour la période exacte, mais {extended_count} images trouvées avec une période étendue de 15 jours")
                collection = extended_collection
                initial_count = extended_count

        collection_with_metrics = collection.map(lambda img: calculate_cloud_and_nodata_metrics(img, geometry))
        filtered_collection = collection_with_metrics \
            .filter(ee.Filter.lte('current_nodata_percentage', max_nodata)) \
            .filter(ee.Filter.lt('current_cloud_percentage', max_cloud_cover))
            
        filtered_count = filtered_collection.size().getInfo()
        
        if feedback:
            feedback.pushInfo(f"Images après filtrage: {filtered_count}")
            feedback.pushInfo(f"Critères: {max_cloud_cover}% nuages max, {max_nodata}% données manquantes max")
            
        if filtered_count == 0:
            relaxed_collection = collection_with_metrics \
                .filter(ee.Filter.lt('current_cloud_percentage', max_cloud_cover))
            relaxed_count = relaxed_collection.size().getInfo()
            
            if relaxed_count == 0:
                raise Exception(f"Aucune image ne correspond aux critères. {initial_count} images initiales.")
            else:
                if feedback:
                    feedback.pushInfo(f"Aucune image ne respecte le critère de données manquantes ({max_nodata}%). Utilisation de {relaxed_count} images avec un filtre relâché.")
                return relaxed_collection
            
        return filtered_collection
        
    except Exception as e:
        error_msg = f"Erreur lors du filtrage: {str(e)}"
        if feedback:
            feedback.pushInfo(f"❌ {error_msg}")
        raise Exception(error_msg)