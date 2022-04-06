import geopandas as gpd
import pandas as pd
import numpy as np
import datetime
import shapely
import osmnx as ox
import mplleaflet
import geo_utils as gu
import create_shap_file as csf

### create point buffer from geodf with given datum ###
def create_point_buffer_wgs84(gdf,dist,proj=None):
    gdf_proj = ox.project_gdf(gdf,to_crs=proj)
    buffer = gpd.GeoDataFrame(geometry=gdf_proj.buffer(dist)).set_crs(gdf_proj.crs)
    buffer_reprojected = ox.project_gdf(buffer,to_crs=gdf.crs)
    buffer_reprojected['place_id'] = buffer_reprojected.index
    return buffer_reprojected

### convert df to geodf ordered by place id###
def gdf_from_df(df,lat_lon_col_names):
    lat,lon = lat_lon_col_names[0], lat_lon_col_names[1]
    gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df[f'{lat}'],df[f'{lon}'])).drop(lat_lon_col_names,axis=1)
    gdf["begin"] = pd.to_datetime(gdf["begin"])
    gdf["end"] = pd.to_datetime(gdf["end"])
    gdf = gdf.sort_values(by=['place_id']).set_crs('EPSG:4326').drop_duplicates(subset=['place_id']).reset_index().drop(columns='index')
    return gdf[['place_id','geometry']]

### check if dwell cluster intersect with given gis layer ###
def cluster_layer_intersection(path,df,dist=40):
    gdf = gdf_from_df(df,['center_longitude','center_latitude'])
    buffers = create_point_buffer_wgs84(gdf,dist)
    gdf_data1 = csf.load_data_in_bbox(f"{path}_polygon.shp",bbox=buffers,all_data=False)
    gdf_data2 = csf.load_data_in_bbox(f"{path}_points.shp",bbox=buffers,all_data=False)
    gdf_data1 = csf.poly_to_centroid(gdf_data1)
    gdf_data = gdf_data2.append(gdf_data1)
    if len(gdf_data) == 0:
        return np.array([]), gdf_data
    else:
        inter = gu.spatial_overlays(buffers, gdf_data, merge_polygon=False)
        places_with_intersection = np.unique(inter.place_id)
        return places_with_intersection, inter.to_crs('EPSG:4326')

















#
