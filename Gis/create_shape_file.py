import osmnx as ox
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import shapely
import geopandas as gpd
import os
import mplleaflet
import basic
import reachability
import networkx as nx
from shapely.geometry import Point


ox.config(log_console=True,use_cache=False)
# ox.config(log_console=True, use_cache=True)

##### makros #######
green_tags = {'leisure':['garden','park','recreation_ground'],
'natural':['scrub','wood','heath','grassland'],
'landuse':['meadow','forest','park','greenfield','village_green','grass','allotments','nature_reserve','recreation_ground','scrub','heath']}
shop_tags ={'shop':['bakery','beverages','convenience','general','department_store','kiosk',
'mall','supermarket','clothes','shoes','chemist','hairdresser','hearing_aids',
'medical_supply','optician','garden_centre','florist','books','gift','pet','laundry','toys']}
poi_tags = {'amenity':['cafe','restaurant','kindergarten','atm','bank','social_facility',
'social_centre','post_office','place_of_worship']}
health = {'amenity':['pharmacy','hospital','doctors','dentist','clinic'],'healthcare':True}
publicTransport_tags = {'highway':['bus_stop','platform'], 'railway':['station','halt','tram_stop']}
citys = ["Havelland, Germany","Berlin, Germany"]

####### save functions #####

### save walk network as gpickle 
def save_streetG(places, network_type='walk',layer_name=f'roads'):
    layer_name += f'_{network_type}'
    for place in places:
        location_name = place.split(',')[0]
        print(f"start with {place}")
        G = ox.graph.graph_from_place(place, network_type=network_type,simplify=False)
        nx.write_gpickle(G,f'roads_{network_type}_{location_name}.pkl')
        
### convert gpickle graph to shapfile
def convert_G(pickle_path,folder_path='Gis_layers',subfolder_path='layer'):
    dir = make_dir(f"{folder_path}/{subfolder_path}")
    G = nx.read_gpickle(pickle_path)
    print("loaded graph start unpacking ...")
    n,e = ox.graph_to_gdfs(G)
    print(f"start saving to disc at {folder_path}/{dir}...")
    n.to_file(f'{dir}/nodes.shp', driver='ESRI Shapefile')
    e.to_file(f'{dir}/edges.shp', driver='ESRI Shapefile')

### save osm data for given places and tags as shapfile
def get_osm_for_place(places,tags,extra=None,layer_name='Max_Gis_layers'):
    place_lst = []
    for place in places:
        print(f"start with {place}")
        tag_lst =['geometry']
        tag_lst.extend(list(tags.keys() ) )
        gdf = ox.geometries.geometries_from_place(place, tags=tags)
        if len(gdf) == 0:
            return gdf
        if 'nodes' in gdf.columns:
            gdf = gdf.reset_index().drop(columns=['osmid','nodes']).copy() #reindex and delete useless data / drop all columns with >1 nan value
        else:
            gdf = gdf.reset_index().drop(columns=['osmid']).copy()
        if extra != None:
            tag_lst.extend(extra)
        gdf = gdf.rename(columns={'element_type': "geom_type"})
        gdf = gdf.set_index('geom_type')
        gdf = gdf.rename(columns={'nature_reserve': "Nreserve"})
        gdf = gdf.rename(columns={'village_green': "VillGreen"})
        gdf = gdf.rename(columns={'recreation_ground': "recreation"})
        gdf = gdf[~ (gdf.geometry.geom_type == 'LineString')]
        place_lst.append(gdf)
    appended_data = pd.concat(place_lst)
    save(layer_name,appended_data,tag_lst)

####### helper #########

### helper to save osm data -> nodes and ways/relations are saved in two different files as requiered by the esri shapfile format
def save(layer_name,gdf,tag_lst):
    dir_name = make_dir()
    point_präfix = 'points'
    polygon_präfix = 'polygon'
    types = set(gdf.index)
    if len(types) == 3:
        points = gdf.loc['node']
        polygons = gdf.loc[['way','relation']]
        points[check_cols(points,tag_lst)].dropna(how='all',axis = 1).to_file(f'{dir_name}/{layer_name}_{point_präfix}.shp', driver='ESRI Shapefile')
        polygons[check_cols(polygons,tag_lst)].dropna(how='all',axis = 1).to_file(f'{dir_name}/{layer_name}_{polygon_präfix}.shp', driver='ESRI Shapefile')
    elif 'node' not in types:
        gdf[check_cols(gdf,tag_lst)].dropna(how='all',axis = 1).to_file(f'{dir_name}/{layer_name}.shp', driver='ESRI Shapefile')
    else:
        points = gdf.loc['node']
        points[check_cols(points,tag_lst)].dropna(how='all',axis = 1).to_file(f'{dir_name}/{layer_name}_{point_präfix}.shp', driver='ESRI Shapefile')
        types.remove('node')
        polygons = gdf.loc["".join(map(str,types))]
        polygons[check_cols(polygons,tag_lst)].dropna(how='all',axis = 1).to_file(f'{dir_name}/{layer_name}_{polygon_präfix}.shp', driver='ESRI Shapefile')

### helper to check tag existence in osm data
def check_cols(gdf, tag_lst):
    tag_set = set(tag_lst)
    columns = [item for item in gdf.columns if item in tag_set]
    return list(columns)

### helper to make dir 
def make_dir(dirName='Gis_layers'):
    try:
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    return dirName

###helper to check if street graph is valid
def is_valid(G):
    if ( (len(G.edges) > 0) & (len(G.nodes) > 0) ):
        return True
    else:
        return False
    
### helper to convert gdf of polygons to center point
def poly_to_centroid(data):
    data = ox.project_gdf(data)
    data['geometry'] = data.centroid
    data = data.to_crs('epsg:4326')
    data.reset_index(inplace=True)
    data['geom_type'] = 'node'
    data = data.set_index('geom_type',drop=True)
    return data

### compose two graphes
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


####### loading functions #####

### load graph in bbox from disc
def load_graph_in_bbox(path,point,dist=2000):
    buff = basic.create_point_buffer(point,dist) #basic aka metric
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

###load osm data around home locaion of the user form disc
def load_data_around_home(path,point,dist=500):
    buff = basic.create_point_buffer(point,dist) #basic aka metric
    buff_proj = buff.to_crs('epsg:4326')
    data = gpd.read_file(path,bbox=buff_proj).set_index('geom_type',drop=True)
    return data

###load osm data in bbox form disc
def load_data_in_bbox(path,bbox=None,all_data=True):
    if all_data == True:
        data = gpd.read_file(path).set_index('geom_type',drop=True)
    else:
        if bbox.crs !=  "epsg:4326":
            bbox = bbox.to_crs('epsg:4326')
        data = gpd.read_file(path,bbox=bbox).set_index('geom_type',drop=True)
    return data

### load graph from from disc in pickle format
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
# G = load_graph_in_bbox('Gis_layers/roads_Havelland',(13.11111, 52.22222))
# data = load_data_around_home('Gis_layers/Green_layer_berlinHavelland_polygon.shp',(13.11111, 52.22222))
# data2 = load_data_in_bbox('Gis_layers/Green_layer_berlinHavelland_polygon.shp',buff,all_data=False)

########################
#### how to ios plot ###
########################
# G = load_graph_in_bbox('Gis_layers/roads_Havelland',(13.11111, 52.22222))
# iso = reachability.reachability_polygon(G,(13.11111, 52.22222),[10],4.5)
# iso.plot()

########################
#### how to compose ####
########################
# G = load_graph_in_bbox('Gis_layers/roads_Berlin',(13.11111, 52.22222))
# G2 = load_graph_in_bbox('Gis_layers/roads_Havelland',(13.11111, 52.22222))
# comp_G = compose(G,G2)


##############################
#### how to combinde data ####
##############################
# data = load_data_around_home('Gis_layers/Shop_layer_berlinHavelland_points.shp',(13.12415, 52.55367))
# data2 = poly_to_centroid(data2)
# comb = data.append(data2)













#
