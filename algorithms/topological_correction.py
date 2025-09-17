# -*- coding: utf-8 -*-
"""
Algorithme de correction topologique - Interface avec le code existant
"""

import os
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterVectorLayer, QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination, QgsProcessingParameterBoolean,
    QgsProcessingParameterCrs, QgsProcessingParameterString,
    QgsProcessingParameterEnum, QgsProcessingException,
    QgsVectorLayer,
    QgsProject,
    QgsProcessingParameterNumber
)
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity, make_valid
import geopandas as gpd
import pandas as pd



class DynamicCulturesParameter(QgsProcessingParameterEnum):
    """Paramètre personnalisé pour la sélection dynamique des cultures"""
    
    def __init__(self, name, description, layer_param_name, field_param_name, optional=True):
        super().__init__(name, description, ['Aucune culture détectée'], allowMultiple=True, optional=optional)
        self.layer_param_name = layer_param_name
        self.field_param_name = field_param_name
    
    def dependsOnOtherParameters(self):
        return [self.layer_param_name, self.field_param_name]


class TopologicalCorrectionAlgorithm(QgsProcessingAlgorithm):
    """Algorithme de correction topologique"""
    
    INPUT_LAYER = 'INPUT_LAYER'
    CULTURE_FIELD = 'CULTURE_FIELD'
    FIX_INVALID = 'FIX_INVALID'
    FIX_OVERLAPS = 'FIX_OVERLAPS'
    FIX_MULTIPART = 'FIX_MULTIPART'
    REMOVE_DUPLICATES = 'REMOVE_DUPLICATES'
    CLEAN_WASTE = 'CLEAN_WASTE'
    REPROJECT = 'REPROJECT'
    TARGET_CRS = 'TARGET_CRS'
    OUTPUT_DIR = 'OUTPUT_DIR'
    WASTE_THRESHOLD = 'WASTE_THRESHOLD' 

    def initAlgorithm(self, config=None):
        """Initialize algorithm parameters"""
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_LAYER,
                self.tr('Couche vectorielle à corriger'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.CULTURE_FIELD,
                self.tr('Champ des types de cultures'),
                parentLayerParameterName=self.INPUT_LAYER,
                type=QgsProcessingParameterField.String
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FIX_INVALID,
                self.tr('Corriger les géométries invalides'),
                defaultValue=True
            )
        )     
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FIX_OVERLAPS,
                self.tr('Corriger les recouvrements'),
                defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FIX_MULTIPART,
                self.tr('Convertir les multiparties'),
                defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.REMOVE_DUPLICATES,
                self.tr('Supprimer les doublons'),
                defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CLEAN_WASTE,
                self.tr('Nettoyer les déchets géométriques'),
                defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.WASTE_THRESHOLD,
                self.tr('Seuil de surface pour les déchets (m²)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=100.0,
                minValue=0.0
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.REPROJECT,
                self.tr('Reprojeter la couche'),
                defaultValue=False
            )
        )
        self.addParameter(
            QgsProcessingParameterCrs(
                self.TARGET_CRS,
                self.tr('Système de coordonnées cible'),
                defaultValue='EPSG:4326',
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR,
                self.tr('Dossier de sortie')
            )
        )
    def ensure_output_directory(self, output_dir, feedback=None):
        """S'assurer que le dossier de sortie existe"""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                if feedback:
                    feedback.pushInfo(f"Dossier de sortie créé: {output_dir}")
            return output_dir
        except Exception as e:
            if feedback:
                feedback.reportError(f"Erreur lors de la création du dossier: {str(e)}")
            raise Exception(f"Impossible de créer le dossier de sortie: {str(e)}")
    
    def processAlgorithm(self, parameters, context, feedback):
        """Process algorithm"""
        
        feedback.pushInfo("="*60)
        feedback.pushInfo("CORRECTION TOPOLOGIQUE DES CULTURES")
        feedback.pushInfo("="*60)
        
        try:
            layer = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)
            culture_field = self.parameterAsString(parameters, self.CULTURE_FIELD, context)
            output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)
            waste_threshold = self.parameterAsDouble(parameters, self.WASTE_THRESHOLD, context) 
            options = {
                'fix_invalid': self.parameterAsBool(parameters, self.FIX_INVALID, context),
                'fix_overlaps': self.parameterAsBool(parameters, self.FIX_OVERLAPS, context),
                'fix_multipart': self.parameterAsBool(parameters, self.FIX_MULTIPART, context),
                'remove_duplicates': self.parameterAsBool(parameters, self.REMOVE_DUPLICATES, context),
                'clean_waste': self.parameterAsBool(parameters, self.CLEAN_WASTE, context),
                'reproject': self.parameterAsBool(parameters, self.REPROJECT, context),
                'waste_threshold': waste_threshold,  
                'target_crs': None
            }
            if options['reproject']:
                target_crs = self.parameterAsCrs(parameters, self.TARGET_CRS, context)
                options['target_crs'] = target_crs.authid()
            
            feedback.pushInfo("Paramètres de traitement:")
            feedback.pushInfo(f"  - Couche: {layer.name()}")
            feedback.pushInfo(f"  - Champ culture: {culture_field}")
            feedback.pushInfo(f"  - Dossier sortie: {output_dir}")
            feedback.pushInfo(f"  - Seuil déchets: {waste_threshold} m²")
            feedback.pushInfo(f"  - Options: {options}")
            correction_stats = {
                'input_file': layer.name(),
                'initial_count': 0,
                'final_count': 0,
                'invalid_geoms_initial': [],
                'initial_area': 0.0,
                'final_area': 0.0,
                'invalid_geoms_final': [],
                'invalid_details': [],
                'overlaps_detected': 0,
                'overlaps_fixed': 0,
                'overlap_removed': 0,
                'overlap_area': 0.0,
                'waste_count': 0,
                'waste_area': 0.0,
                'waste_details': [],
                'duplicates_removed': 0,
                'duplicate_ids': [],
                'multipart_converted': 0,
                'multipart_details': []
            }
            if layer.featureCount() == 0:
                raise QgsProcessingException("La couche sélectionnée ne contient aucune entité")
            def layer_to_geodataframe(layer):
                try:
                    features_data = []
                    for feature in layer.getFeatures():
                        geom = feature.geometry()
                        if geom and not geom.isEmpty():
                            wkt = geom.asWkt()
                            attributes = feature.attributes()
                            field_names = [field.name() for field in layer.fields()]
                            feature_dict = dict(zip(field_names, attributes))
                            feature_dict['geometry'] = wkt
                            features_data.append(feature_dict)
                    
                    if not features_data:
                        return None
                    df = pd.DataFrame(features_data)
                    from shapely import wkt
                    df['geometry'] = df['geometry'].apply(wkt.loads)
                    gdf = gpd.GeoDataFrame(df, geometry='geometry')
                    if layer.crs().isValid():
                        gdf.crs = layer.crs().authid()
                    
                    return gdf
                    
                except Exception as e:
                    feedback.reportError(f"Erreur lors de la conversion: {str(e)}")
                    return None
            gdf = layer_to_geodataframe(layer)
            if gdf is None or len(gdf) == 0:
                raise QgsProcessingException("Impossible de lire les entités de la couche")
            correction_stats['initial_count'] = len(gdf)
            feedback.pushInfo(f"Fichier chargé avec succès : {len(gdf)} entités")
            feedback.setProgress(5)
            if culture_field not in gdf.columns:
                raise QgsProcessingException(f"Le champ '{culture_field}' n'existe pas")
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
            if gdf.crs is None:
                if layer.crs().isValid():
                    gdf.set_crs(layer.crs().authid(), inplace=True)
                    feedback.pushInfo(f"CRS défini à partir de la couche: {layer.crs().authid()}")
                else:
                    raise QgsProcessingException("Impossible de déterminer le CRS")
            
            if options.get('reproject', False) and options.get('target_crs'):
                target_crs = options['target_crs']
                if gdf.crs.to_string() != target_crs:
                    feedback.pushInfo(f"Reprojection de {gdf.crs.to_string()} vers {target_crs}")
                    gdf = gdf.to_crs(target_crs)
                    feedback.pushInfo("Reprojection terminée avec succès")
                else:
                    feedback.pushInfo("La couche utilise déjà le CRS cible")
            else:
                feedback.pushInfo(f"Conservation du CRS original: {gdf.crs.to_string()}")

            correction_stats['initial_area'] = gdf.geometry.area.sum()
            feedback.pushInfo(f"Nombre total d'entités avant traitement : {len(gdf)}")
            feedback.pushInfo(f"Surface totale initiale : {correction_stats['initial_area']:.2f} m²")
            
            feedback.setProgress(10)
            
            # =========================================================================
            # PARTIE 1: GÉOMÉTRIE INVALIDE 
            # =========================================================================
            if options.get('fix_invalid', True):
                feedback.pushInfo("="*40)
                feedback.pushInfo("DÉBUT DU TRAITEMENT DE LA GÉOMÉTRIE INVALIDE")
                buffer_distance = 5
                gdf['geometry'] = gdf['geometry'].buffer(-buffer_distance, join_style=2).buffer(buffer_distance, join_style=2)
                
                gdf['is_valid'] = gdf.geometry.is_valid
                gdf['err_reason'] = gdf.geometry.apply(explain_validity)
                gdf['idPY'] = gdf.index
                
                initial_invalids = gdf[~gdf['is_valid']].copy()
                invalid_ids = initial_invalids['idPY'].tolist()
                feedback.pushInfo(f"Géométries invalides initiales (ID): {invalid_ids}")
                feedback.pushInfo(f"{len(initial_invalids)} géométries invalides détectées initialement.")
                
                for idx, row in initial_invalids.iterrows():
                    correction_stats['invalid_details'].append({
                        'id': row['idPY'],
                        'error_type': row['err_reason'],
                        'culture': row[culture_field],
                        'area': row.geometry.area
                    })
                
                gdf["area_befor"] = gdf.geometry.area
                
                def robust_geometry_fixer(geom):
                    if geom.is_valid:
                        return geom
                    buffered = geom.buffer(0)
                    if buffered.is_valid:
                        return buffered
                    repaired = make_valid(geom)
                    if repaired.is_valid:
                        return repaired
                    if repaired.geom_type == 'GeometryCollection':
                        polygons = [g for g in repaired.geoms if isinstance(g, (Polygon, MultiPolygon))]
                        if polygons:
                            return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
                    return None
                
                gdf['geometry'] = gdf.geometry.apply(robust_geometry_fixer)
                gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
                gdf['is_valid_f'] = gdf.geometry.is_valid
                final_invalids = gdf[~gdf['is_valid_f']]
                fixed_ids = list(set(initial_invalids['idPY']) - set(final_invalids['idPY']))
                feedback.pushInfo(f"Géométries corrigées (ID): {fixed_ids}")
                feedback.pushInfo(f"{len(initial_invalids)-len(final_invalids)} géométries corrigées")
                
                remaining_invalids = final_invalids['idPY'].tolist()
                feedback.pushInfo(f"Géométries toujours invalides (ID): {remaining_invalids}")
                
                gdf['area_after'] = gdf.geometry.area
                gdf['delta_area'] = gdf['area_after'] - gdf['area_befor']
                
                feedback.pushInfo("Résumé des modifications:")
                feedback.pushInfo(f"- Surface totale avant : {gdf['area_befor'].sum():.2f} m²")
                feedback.pushInfo(f"- Surface totale après : {gdf['area_after'].sum():.2f} m²")
                feedback.pushInfo(f"- Variation totale : {gdf['delta_area'].sum():.2f} m²")
                
                feedback.setProgress(25)
            
            # =========================================================================
            # PARTIE 2: RECOUVREMENT 
            # =========================================================================
            if options.get('fix_overlaps', True):
                feedback.pushInfo("="*40)
                feedback.pushInfo("DÉBUT DU TRAITEMENT DES RECOUVREMENTS")
                feedback.pushInfo("="*40)
                
                gdf['idPY'] = gdf.index
                total_overlap_area = 0.0
                total_overlaps_detected = 0
                total_overlaps_fixed = 0
                intersections_supprimees = []
                to_remove = set()
                
                sindex = gdf.sindex
                
                for idx in gdf.index:
                    geom = gdf.at[idx, 'geometry']
                    if geom is None or not geom.is_valid:
                        continue
                    
                    possible_matches_index = list(sindex.intersection(geom.bounds))
                    voisins = [i for i in possible_matches_index if i > idx]
                    
                    for j in voisins:
                        if j in to_remove or idx in to_remove:
                            continue
                        if j not in gdf.index:
                            continue
                        neigh_geom = gdf.at[j, 'geometry']
                        if neigh_geom is None or not neigh_geom.is_valid:
                            continue
                            
                        if not geom.overlaps(neigh_geom):
                            continue
                            
                        inter = geom.intersection(neigh_geom)
                        if inter.is_empty or not inter.is_valid:
                            continue
                            
                        if inter.geom_type in ['Polygon', 'MultiPolygon']:
                            intersections_supprimees.append({
                                "id_1": idx,
                                "id_2": j,
                                "geometry": inter
                            })
                        
                        area = inter.area
                        total_overlap_area += area
                        total_overlaps_detected += 1
                        ratio_i = inter.area / geom.area
                        ratio_j = inter.area / neigh_geom.area
                        
                        cult_i = gdf.at[idx, culture_field]
                        cult_j = gdf.at[j, culture_field]
                        
                        
                        if cult_i != cult_j:
                            if ratio_i > 0.6 and ratio_j > 0.6:
                                to_remove.update([idx, j])
                                feedback.pushInfo(f"Suppression des entités {idx} et {j} (>60% recouvrement mutuel, cultures différentes)")
                            elif ratio_i > 0.6:
                                to_remove.add(idx)
                                new_j = neigh_geom.difference(inter)
                                if not new_j.is_empty and new_j.is_valid:
                                    gdf.at[j, 'geometry'] = new_j
                                feedback.pushInfo(f"Suppression entité {idx} (>60%), ajustement {j}")
                            elif ratio_j > 0.6:
                                to_remove.add(j)
                                new_i = geom.difference(inter)
                                if not new_i.is_empty and new_i.is_valid:
                                    gdf.at[idx, 'geometry'] = new_i
                                feedback.pushInfo(f"Suppression entité {j} (>60%), ajustement {idx}")
                            else:
                                new_i = geom.difference(inter)
                                new_j = neigh_geom.difference(inter)
                                if not new_i.is_empty and new_i.is_valid:
                                    gdf.at[idx, 'geometry'] = new_i
                                if not new_j.is_empty and new_j.is_valid:
                                    gdf.at[j, 'geometry'] = new_j
                                feedback.pushInfo(f"Ajustement standard des entités {idx} et {j}")
                        else:
                            if ratio_i > 0.6 and ratio_j > 0.6:
                                union = geom.union(neigh_geom)
                                if not union.is_empty and union.is_valid:
                                    gdf.at[idx, 'geometry'] = union
                                    to_remove.add(j)
                                feedback.pushInfo(f"Fusion des entités {idx} et {j} (même culture, >60% recouvrement)")
                            elif ratio_i > 0.6:
                                to_remove.add(idx)
                                feedback.pushInfo(f"Suppression entité {idx} (même culture, >60% recouvrement)")
                            elif ratio_j > 0.6:
                                to_remove.add(j)
                                feedback.pushInfo(f"Suppression entité {j} (même culture, >60% recouvrement)")
                            else:
                                new_i = geom.difference(inter)
                                if not new_i.is_empty and new_i.is_valid:
                                    gdf.at[idx, 'geometry'] = new_i
                                feedback.pushInfo(f"Ajustement standard entité {idx} (même culture)")
                        
                        total_overlaps_fixed += 1
                
                if to_remove:
                    gdf = gdf.drop(index=list(to_remove))
                    correction_stats['overlap_removed'] = len(to_remove)
                    feedback.pushInfo(f"{len(to_remove)} entités supprimées pour recouvrement excessif")
                correction_stats['overlaps_detected'] = total_overlaps_detected
                correction_stats['overlaps_fixed'] = total_overlaps_fixed
                correction_stats['overlap_area'] = total_overlap_area

                if intersections_supprimees:
                    base_name = layer.name().replace(' ', '_')
                    self.ensure_output_directory(output_dir, feedback)
                    intersections_output = os.path.join(output_dir, f"{base_name}_intersections.shp")
                    try:
                        def convert_geometry(geom):
                            if geom.geom_type == 'GeometryCollection':
                                polygons = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
                                return MultiPolygon(polygons) if polygons else None
                            return geom
                        
                        gdf_intersections = gpd.GeoDataFrame(intersections_supprimees, crs=gdf.crs)
                        if not gdf_intersections.empty:
                            gdf_intersections['geometry'] = gdf_intersections['geometry'].apply(convert_geometry)
                            gdf_intersections = gdf_intersections.dropna(subset=['geometry'])
                        
                        if not gdf_intersections.empty:
                            gdf_intersections.to_file(intersections_output)
                            feedback.pushInfo(f"{len(gdf_intersections)} zones de recouvrement exportées: {intersections_output}")
                        else:
                            feedback.pushInfo("Aucune intersection valide trouvée")
                    except Exception as e:
                        feedback.reportError(f"Erreur lors de l'exportation des intersections: {str(e)}")               
                def corriger_recouvrements_precis(gdf):
                    total_overlap_area2 = 0
                    gdf = gdf.reset_index(drop=True)
                    geometries_corrigees = []
                    
                    for idx, row in gdf.iterrows():
                        geom = row.geometry
                        autres = gdf.drop(idx)
                        
                        for autre_geom in autres.geometry:
                            if geom.overlaps(autre_geom):
                                inter2 = geom.intersection(autre_geom)
                                if not inter2.is_empty:
                                    total_overlap_area2 += inter2.area
                                    geom = geom.difference(autre_geom)
                        geometries_corrigees.append(geom)
                    
                    feedback.pushInfo(f"Surface totale des Recouvrements réstants : {total_overlap_area2:.2f} m²")
                    gdf["geometry"] = geometries_corrigees
                    return gdf
                
                gdf = corriger_recouvrements_precis(gdf)
                gdf['geometry'] = gdf['geometry'].buffer(-0.001, join_style=2)
                
                
                gdf['area_after'] = gdf.geometry.area
                gdf['delta_area'] = gdf['area_after'] - gdf['area_befor']
                
                feedback.pushInfo("Statistiques des recouvrements:")
                feedback.pushInfo(f"- Recouvrements détectés : {total_overlaps_detected}")
                feedback.pushInfo(f"- Recouvrements corrigés : {total_overlaps_fixed}")
                feedback.pushInfo(f"- Taux de correction : {total_overlaps_fixed/max(1,total_overlaps_detected)*100:.2f}%")
                feedback.pushInfo(f"- Surface totale des recouvrements : {total_overlap_area:.2f} m²")
                
                feedback.pushInfo("Résumé final des modifications:")
                feedback.pushInfo(f"- Surface totale avant : {gdf['area_befor'].sum():.2f} m²")
                feedback.pushInfo(f"- Surface totale après : {gdf['area_after'].sum():.2f} m²")
                feedback.pushInfo(f"- Variation totale : {gdf['delta_area'].sum():.2f} m²")
                
                feedback.setProgress(50)
            
            # =========================================================================
            # PARTIE 3: MULTIPARTIES 
            # =========================================================================
            if options.get('fix_multipart', True):
                feedback.pushInfo("="*40)
                feedback.pushInfo("DÉBUT DU TRAITEMENT MULTIPARTIES")
                feedback.pushInfo("="*40)
                
                feedback.pushInfo(f"Nombre total d'entités avant explode : {len(gdf)}")
                
                multipart_count = sum(gdf.geometry.type == 'MultiPolygon')
                feedback.pushInfo(f"Nombre d'entités multipart : {multipart_count}")
                multipart_geoms = gdf[gdf.geometry.type == 'MultiPolygon']
                for idx, row in multipart_geoms.iterrows():
                    part_count = len(row.geometry.geoms) if hasattr(row.geometry, 'geoms') else 1
                    correction_stats['multipart_details'].append({
                        'id': row['idPY'],
                        'parts_count': part_count,
                        'area': row.geometry.area
                    })
                
                gdf_single = gdf.explode(index_parts=True).reset_index(drop=True)
                feedback.pushInfo(f"Nombre total d'entités après explode : {len(gdf_single)}")
                
                geom_types = gdf_single.geometry.type.unique()
                feedback.pushInfo(f"Types de géométries après conversion : {', '.join(geom_types)}")
                
                correction_stats['multipart_converted'] = multipart_count
                gdf = gdf_single
                
                feedback.setProgress(70)
            
            # =========================================================================
            # PARTIE 4: DOUBLONS 
            # =========================================================================
            if options.get('remove_duplicates', True):
                feedback.pushInfo("="*40)
                feedback.pushInfo("DÉBUT DE LA DÉTECTION DES DOUBLONS")
                feedback.pushInfo("="*40)
                
                doublons_exacts = gdf[gdf.duplicated(subset='geometry')]
                
                if not doublons_exacts.empty:
                    feedback.pushInfo("Doublons exacts détectés :")
                    
                    for dup in doublons_exacts.itertuples():
                        cult = getattr(dup, culture_field)
                        feedback.pushInfo(f"ID: {dup.idPY}, Culture: {cult}, Surface: {dup.geometry.area:.2f} m²")
                        correction_stats['duplicate_ids'].append(dup.idPY)
                else:
                    feedback.pushInfo("Aucun doublon exact trouvé.")
                
                initial_count = len(gdf)
                gdf = gdf.drop_duplicates(subset='geometry')
                duplicates_removed = initial_count - len(gdf)
                correction_stats['duplicates_removed'] = duplicates_removed
                feedback.pushInfo(f"Nombre d'entités après suppression des doublons : {len(gdf)} ({duplicates_removed} supprimés)")
                
                feedback.setProgress(80)
            
            # =========================================================================
            # PARTIE 5: Nettoyage des déchets géométriques 
            # =========================================================================
            if options.get('clean_waste', True):
                feedback.pushInfo("="*40)
                feedback.pushInfo("DÉBUT DU NETTOYAGE DES DÉCHETS GÉOMÉTRIQUES")
                feedback.pushInfo("="*40)
                feedback.pushInfo(f"Utilisation du seuil de {waste_threshold} m²")
                
                deleted_areas = []
                total_waste_area = 0.0
                waste_details = []
                
                def clean_geometries(geom):
                    nonlocal total_waste_area, waste_details, deleted_areas
                    if geom is None or geom.is_empty:
                        return None
                    if geom.geom_type == 'Polygon':
                        if geom.area >= waste_threshold:
                            return geom
                        else:
                            deleted_areas.append(geom.area)
                            total_waste_area += geom.area
                            waste_details.append({
                                'type': 'Polygone',
                                'area': geom.area,
                                'reason': f'Surface trop petite (<{waste_threshold} m²)'
                            })
                            return None
                    elif geom.geom_type == 'MultiPolygon':
                        kept = []
                        for p in geom.geoms:
                            if p.area >= waste_threshold:
                                kept.append(p)
                            else:
                                deleted_areas.append(p.area)
                                total_waste_area += p.area
                                waste_details.append({
                                    'type': 'Partie de MultiPolygone',
                                    'area': p.area,
                                    'reason': f'Surface trop petite (<{waste_threshold} m²)'
                                })
                        return MultiPolygon(kept) if kept else None
                    else:
                        waste_details.append({
                            'type': geom.geom_type,
                            'area': geom.area if hasattr(geom, 'area') else 0,
                            'reason': 'Type de géométrie non supporté'
                        })
                        return None
                
                initial_count = len(gdf)
                gdf['geometry'] = gdf['geometry'].apply(clean_geometries)
                gdf = gdf[gdf.geometry.notna()]
                
                deleted_count = initial_count - len(gdf)
                correction_stats['waste_count'] = deleted_count
                correction_stats['waste_area'] = total_waste_area
                correction_stats['waste_details'] = waste_details
                
                feedback.pushInfo(f"- {deleted_count} déchets géométriques supprimés")
                feedback.pushInfo(f"- {len(gdf)} géométries conservées après nettoyage")
                feedback.pushInfo(f"- Surface totale des déchets : {total_waste_area:.2f} m²")
                
                if waste_details:
                    feedback.pushInfo("Détail des déchets supprimés :")
                    for i, waste in enumerate(waste_details, 1):
                        feedback.pushInfo(f"Déchet {i}: Type={waste['type']}, Surface={waste['area']:.2f} m², Raison={waste['reason']}")
                else:
                    feedback.pushInfo("Aucun déchet à afficher")
                gdf = gdf[
                    gdf.geometry.notna() & 
                    ~gdf.geometry.is_empty & 
                    gdf.geometry.is_valid
                ]
                
                def robust_geometry_fixer(geom):
                    if geom.is_valid:
                        return geom
                    buffered = geom.buffer(0)
                    if buffered.is_valid:
                        return buffered
                    repaired = make_valid(geom)
                    if repaired.is_valid:
                        return repaired
                    if repaired.geom_type == 'GeometryCollection':
                        polygons = [g for g in repaired.geoms if isinstance(g, (Polygon, MultiPolygon))]
                        if polygons:
                            return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
                    return None
                
                if 'is_valid_post_snap' in gdf.columns:
                    invalid_idx = gdf[~gdf['is_valid_post_snap']].index
                    if not invalid_idx.empty:
                        gdf.loc[invalid_idx, 'geometry'] = (
                            gdf.loc[invalid_idx, 'geometry']
                            .apply(robust_geometry_fixer)
                        )
                        gdf = gdf[gdf.geometry.is_valid]
                
                feedback.pushInfo(f"Après réparation : {len(gdf)} géométries valides")
                
                geom_types = gdf.geometry.type.unique()
                feedback.pushInfo(f"Types de géométries après conversion : {', '.join(geom_types)}")
                
                feedback.setProgress(90)
            
            correction_stats['final_count'] = len(gdf)
            correction_stats['final_area'] = gdf.geometry.area.sum()

            feedback.pushInfo("Sauvegarde des résultats...")
            self.ensure_output_directory(output_dir, feedback)
            base_name = layer.name().replace(' ', '_')
            output_path = os.path.join(output_dir, f"{base_name}_corrige.shp")
            try:
                gdf.to_file(output_path)
                feedback.pushInfo(f"Couche corrigée sauvegardée: {output_path}")
            except Exception as e:
                feedback.reportError(f"Erreur lors de la sauvegarde: {str(e)}")
                raise QgsProcessingException(f"Impossible de sauvegarder le fichier: {str(e)}")
            
            def generate_detailed_report():
                try:
                    self.ensure_output_directory(output_dir, feedback)
                    
                    rapport = "\n" + "="*80
                    rapport += "\nRAPPORT FINAL DES CORRECTIONS - DÉTAILLÉ"
                    rapport += "\n" + "="*80
                    
                    rapport += f"\n\nFICHIER TRAITÉ: {correction_stats['input_file']}"
                    rapport += f"\nEntités initiales: {correction_stats['initial_count']}"
                    rapport += f"\nEntités finales: {correction_stats['final_count']}"
                    rapport += f"\nSurface initiale: {correction_stats['initial_area']:.2f} m²"
                    rapport += f"\nSurface finale: {correction_stats['final_area']:.2f} m²"
                    rapport += f"\nDifférence de surface: {correction_stats['final_area'] - correction_stats['initial_area']:.2f} m²"
                    
                    rapport += "\n\nGÉOMÉTRIES INVALIDES:"
                    rapport += f"\n- Initiales: {len(correction_stats['invalid_details'])} entités"
                    if correction_stats['invalid_details']:
                        rapport += "\n  Détail des erreurs:"
                        for detail in correction_stats['invalid_details']:
                            rapport += f"\n  - ID {detail['id']}: {detail['error_type']} (Culture: {detail['culture']}, Surface: {detail['area']:.2f} m²)"
                    
                    rapport += "\n\nRECOUVREMENTS:"
                    rapport += f"\n- Détectés: {correction_stats['overlaps_detected']}"
                    rapport += f"\n- Corrigés: {correction_stats['overlaps_fixed']}"
                    rapport += f"\n- Entités supprimées (recouvrement excessif): {correction_stats['overlap_removed']}"
                    rapport += f"\n- Surface totale des recouvrements: {correction_stats['overlap_area']:.2f} m²"
                    
                    rapport += "\n\nDÉCHETS GÉOMÉTRIQUES:"
                    rapport += f"\n- Éléments supprimés: {correction_stats['waste_count']}"
                    rapport += f"\n- Surface totale des déchets: {correction_stats['waste_area']:.2f} m²"
                    
                    if correction_stats['waste_details']:
                        rapport += "\n  Détail des déchets:"
                        for i, waste in enumerate(correction_stats['waste_details'], 1):
                            rapport += f"\n  - Déchet {i}: Type={waste['type']}, Surface={waste['area']:.2f} m², Raison={waste['reason']}"
                    
                    rapport += "\n\nDOUBLONS:"
                    rapport += f"\n- Supprimés: {correction_stats['duplicates_removed']}"
                    if correction_stats['duplicate_ids']:
                        rapport += f"\n- IDs des doublons supprimés: {', '.join(map(str, correction_stats['duplicate_ids']))}"
                    
                    rapport += "\n\nMULTIPARTIES:"
                    rapport += f"\n- Entités converties: {correction_stats['multipart_converted']}"
                    if correction_stats['multipart_details']:
                        rapport += "\n  Détail des multiparties:"
                        for detail in correction_stats['multipart_details']:
                            rapport += f"\n  - ID {detail['id']}: {detail['parts_count']} parties, Surface: {detail['area']:.2f} m²"
                    
                    rapport += "\n\n---\nPlugin développé par Hiba Aamnous dans le cadre de PFE"
                    rapport += "\nPôle Digital de l'Agriculture, de la Forêt et Observatoire de la Sécheresse - 2025"
                    rapport += "\n\n" + "="*80 + "\nFIN DU RAPPORT DÉTAILLÉ\n" + "="*80
                    
                    report_path = os.path.join(output_dir, f"Rapport_Detaille_{base_name}.txt")
                    
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(rapport)
                    
                    feedback.pushInfo(f"Rapport détaillé généré: {report_path}")
                    
                except Exception as e:
                    feedback.reportError(f"Erreur lors de la génération du rapport: {str(e)}")
            
            generate_detailed_report()
            corrected_layer = QgsVectorLayer(output_path, f"{base_name}_corrige", "ogr")
            if corrected_layer.isValid():
                QgsProject.instance().addMapLayer(corrected_layer)
                feedback.pushInfo(f"Couche corrigée ajoutée au projet: {base_name}_corrige")
            
            feedback.pushInfo("="*60)
            feedback.pushInfo("CORRECTION TOPOLOGIQUE TERMINÉE AVEC SUCCÈS")
            feedback.pushInfo("="*60)
            
            return {self.OUTPUT_DIR: output_dir}
            
        except Exception as e:
            feedback.reportError(f"ERREUR: {str(e)}")
            raise QgsProcessingException(str(e))

    def name(self):
        return 'topological_correction'

    def displayName(self):
        return self.tr('5 - Correction Topologique')

    def group(self):
        return self.tr('Cartographie des Cultures')

    def groupId(self):
        return 'agriculture_mapping'

    def shortHelpString(self):
        return self.tr("""
        <h3>Correction Topologique des Couches Vectorielles</h3>
        <p>Corrige automatiquement les erreurs topologiques dans les données de cartographie des cultures.</p>
        
        <h4>Corrections disponibles:</h4>
        <ul>
        <li><b>Géométries invalides:</b> Auto-intersections, boucles, nœuds</li>
        <li><b>Recouvrements:</b> Résolution intelligente selon les cultures</li>
        <li><b>Multi-parties:</b> Conversion en géométries simples</li>
        <li><b>Doublons:</b> Suppression des géométries identiques</li>
        <li><b>Déchets géométriques:</b> Nettoyage des micro-polygones (seuil configurable)</li>
        </ul>
        
        <h4>Options avancées:</h4>
        <li><b>Reprojection:</b> Changement de système de coordonnées</li>
        <li><b>Seuil déchets:</b> Spécification de la surface minimale à conserver</li>
        </ul>
        
        <h4>Utilisation:</h4>
        <p>Utilisez cet algorithme depuis la boîte à outils Processing pour corriger automatiquement les erreurs topologiques de vos données de cartographie des cultures.</p>
        
        <h4>Sorties:</h4>
        <ul>
        <li><b>Couche corrigée:</b> Shapefile sans erreurs topologiques</li>
        <li><b>Rapports détaillés:</b> Statistiques des corrections</li>
        <li><b>Zones de recouvrement:</b> Shapefile des intersections détectées</li>
        </ul>
        """)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def icon(self):
        """Return algorithm icon"""
        from ..help_system import HelpSystem
        help_system = HelpSystem(os.path.dirname(os.path.dirname(__file__)))
        return help_system.get_algorithm_icon('topological_correction')

    def helpUrl(self):
        """Return help URL"""
        from ..help_system import HelpSystem
        help_system = HelpSystem(os.path.dirname(os.path.dirname(__file__)))
        return help_system.get_help_url(self.name())

    def createInstance(self):
        return TopologicalCorrectionAlgorithm()

