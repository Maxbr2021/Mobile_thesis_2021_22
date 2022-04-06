import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import shapely
import geopandas as gpd
from shapely.geometry import Point


##### helper ######
def is_valid(G):
    if ( (len(G.edges) > 0) & (len(G.nodes) > 0) ):
        return True
    else:
        return False

def create_point_buffer(lon_lat,dist,proj=None):
    point = build_geo_df(lon_lat)
    point_proj = ox.project_gdf(point,to_crs=proj)
    buffer = gpd.GeoDataFrame(geometry=point_proj.buffer(dist)).set_crs(point_proj.crs)
    return buffer

def poly_to_centroid(data):
    try:
        data = ox.project_gdf(data)
        data['geometry'] = data.centroid
        data = data.to_crs('epsg:4326')
        data.reset_index(inplace=True)
        data['geom_type'] = 'node'
        data = data.set_index('geom_type',drop=True)
        return data
    except:
        return data

def compose(G,G2):
    if (G and G2) != None:
        composed = nx.compose(G,G2)
    elif G != None:
        composed = G
    elif G2 != None:
        composed = G2
    else:
        print("error no graph is valid")
        composed = None
    return composed

def to_point(lon_lat):
    ### convert array to point ###
    return shapely.geometry.Point(lon_lat[0],lon_lat[1])

def build_geo_df(lon_lat):
    #build geoDatafram for point of interest
    point = to_point(lon_lat)
    geo_s = gpd.GeoSeries(point,dtype='object')
    geo_df = gpd.GeoDataFrame(geometry=geo_s)
    geo_df = geo_df.set_crs('epsg:4326') #set lon lat input to wsg84 crs
    return geo_df

def calc_ratio(buffer,inter):
    #calculate the ratio of <inter> in the buffer zone
    return round((inter.unary_union.area / buffer.unary_union.area) * 100,2) #NOTE: inter is merged to one polygon due to overlapping prevention

def spatial_overlays(buffer,geo_df,how='intersection',merge_polygon=True):
    if geo_df.crs == None or geo_df.crs != buffer.crs:
        geo_df_proj = ox.project_gdf(geo_df,to_crs=buffer.crs) #make sure both are in the same crs
    else:
        geo_df_proj = geo_df
    if merge_polygon == True:
        overlay = gpd.overlay(buffer,polygon_to_gdf(geo_df_proj,geo_df_proj.crs),how=how,keep_geom_type=False)
    else:
        overlay = gpd.overlay(buffer,geo_df_proj,how=how,keep_geom_type=False) # only use if df2 are same geomety types for sure
    return overlay

## helper for overlay combine gdf to unary_union polygon and return as gdf
def polygon_to_gdf(gdf_p,proj=None):
    polygon = gdf_p.unary_union
    s = gpd.GeoSeries(polygon)
    gdf = gpd.GeoDataFrame(geometry=s).set_crs(proj)
    return gdf

####### loading functions #####

def load_graph_in_bbox(path,point,dist=1000):
    buff = create_point_buffer(point,dist)
    buff_proj = buff.to_crs("epsg:4326")
    nodes = gpd.read_file(f"{path}/nodes.shp",bbox=buff_proj).set_index('osmid',drop=True)
    edges = gpd.read_file(f"{path}/edges.shp",bbox=buff_proj).set_index(['u','v','key'],drop=True)
    G = ox.utils_graph.graph_from_gdfs(nodes, edges)
    if is_valid(G):
        node,data = zip(*G.nodes(data=True))
        newdata = []
        geom=[]
        newn = []
        for i,d in enumerate(data):
            if bool(d):
                newn.append(node[i])
                newdata.append(d)
                geom.append(Point(d['x'],d['y']) )
        newdata = tuple(newdata)
        gdf_nodes = gpd.GeoDataFrame(newdata,index=newn,crs='epsg:4326',geometry=geom)
        gdf_nodes.index.rename("osmid", inplace=True)
        i = gdf_nodes.index
        sub = G.subgraph(i)
        if is_valid(sub):
            return sub
        else:
            return None
    else:
        return None

def load_graph_from_pickle(home,dist=1000):
    G = nx.read_gpickle('Gis_layers/roads_walk_Berlin_Havelland.pkl')
    nodes = ox.graph_to_gdfs(G, edges=False)
    buff = create_point_buffer(home,dist) #basic aka metric
    buff_proj = buff.to_crs("epsg:4326")
    intersecting_nodes = nodes[nodes.intersects(buff_proj['geometry'][0])].index
    G_sub = G.subgraph(intersecting_nodes)
    del G
    if is_valid(G_sub):
        G_simple = ox.simplification.simplify_graph(G_sub)
        buff_area = buff['geometry'][0].area
        return_tuple =  (ox.utils_graph.get_largest_component(G_sub), ox.basic_stats(G_simple,buff_area) )
        return_tuple[1]["intersection_3_way_density_km"] = ox.stats.intersection_count(G=G_simple, min_streets=3) / (buff_area / 1_000_000)
        return return_tuple
    else:
        return None, None

def load_data_around_home(path,point,dist=500):
    buff = create_point_buffer(point,dist)
    buff_proj = buff.to_crs('epsg:4326')
    data = gpd.read_file(path,bbox=buff_proj).set_index('geom_type',drop=True)
    return data

def load_data_in_bbox(path,bbox=None,all_data=True):
    if all_data == True:
        data = gpd.read_file(path).set_index('geom_type',drop=True)
    else:
        if bbox.crs !=  "epsg:4326":
            bbox = bbox.to_crs('epsg:4326')
        data = gpd.read_file(path,bbox=bbox).set_index('geom_type',drop=True)
    return data

#################################
#### how to download and save ###
#################################
# G = save_streetG(citys)
# convert_G('roads_walk_Havelland.pkl',subfolder_path='roads_Havelland')
# get_osm_for_place(citys,green_tags,layer_name='Green_layer_berlinHavelland')
# convert_G('roads_walk_Berlin.pkl',subfolder_path='roads_Berlin')

########################
#### how to load #######
########################
# G = load_graph_in_bbox('Gis_layers/roads_Havelland',(13.12415, 52.55367))
# data = load_data_around_home('Gis_layers/Green_layer_berlinHavelland_polygon.shp',(13.12415, 52.55367))
# data2 = load_data_in_bbox('Gis_layers/Green_layer_berlinHavelland_polygon.shp',buff,all_data=False)

########################
#### how to ios plot ###
########################
# G = load_graph_in_bbox('Gis_layers/roads_Havelland',(13.12415, 52.55367))
# iso = reachability.reachability_polygon(G,(13.12415, 52.55367),[10],4.5)
# iso.plot()

########################
#### how to compose ####
########################
# G = load_graph_in_bbox('Gis_layers/roads_Berlin',(13.12415, 52.55367))
# G2 = load_graph_in_bbox('Gis_layers/roads_Havelland',(13.12415, 52.55367))
# comp_G = compose(G,G2)


##############################
#### how to combinde data ####
##############################
# data = load_data_around_home('Gis_layers/Shop_layer_berlinHavelland_points.shp',(13.12415, 52.55367))
# data2 = poly_to_centroid(data2)
# comb = data.append(data2)
