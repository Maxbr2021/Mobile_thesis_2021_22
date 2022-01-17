import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from descartes import PolygonPatch
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

### build reachability polygon ###
def make_iso_polys(G, center_node,trip_times, edge_buff=25, node_buff=50, infill=False):
    # NOTE: function from https://github.com/gboeing/osmnx-examples/blob/main/notebooks/13-isolines-isochrones.ipynb
    isochrone_polys = []
    for trip_time in sorted(trip_times, reverse=True):
        subgraph = nx.ego_graph(G, center_node, radius=trip_time, distance="time")

        node_points = [Point((data["x"], data["y"])) for node, data in subgraph.nodes(data=True)]
        nodes_gdf = gpd.GeoDataFrame({"id": list(subgraph.nodes)}, geometry=node_points)
        nodes_gdf = nodes_gdf.set_index("id")

        edge_lines = []
        for n_fr, n_to in subgraph.edges():
            f = nodes_gdf.loc[n_fr].geometry
            t = nodes_gdf.loc[n_to].geometry
            edge_lookup = G.get_edge_data(n_fr, n_to)[0].get("geometry", LineString([f, t]))
            edge_lines.append(edge_lookup)

        n = nodes_gdf.buffer(node_buff).geometry
        e = gpd.GeoSeries(edge_lines).buffer(edge_buff).geometry
        all_gs = list(n) + list(e)
        new_iso = gpd.GeoSeries(all_gs).unary_union

        # try to fill in surrounded areas so shapes will appear solid and
        # blocks without white space inside them
        if infill:
            new_iso = Polygon(new_iso.exterior)
        isochrone_polys.append(new_iso)
    return isochrone_polys

### build reachability polygon geo df ###
def reachability_polygon(g,point,trip_times,travel_speed,to_crs=None,edge_buff=25, node_buff=50, infill=False,simplify=False):
    # NOTE: function based of https://github.com/gboeing/osmnx-examples/blob/main/notebooks/13-isolines-isochrones.ipynb
    #G = ox.graph_from_point(point[::-1],1000, network_type='walk',simplify=simplify)
    G=g
    print('G')
    center_node = ox.distance.nearest_nodes(G, point[0], point[1])
    G = ox.project_graph(G,to_crs=to_crs)
    print('proj')

    # add an edge attribute for time in minutes required to traverse each edge
    meters_per_minute = travel_speed * 1000 / 60  # km per hour to m per minute
    for _, _, _, data in G.edges(data=True, keys=True):
        data["time"] = data["length"] / meters_per_minute
    isochrone_polys = make_iso_polys(G,center_node,trip_times, edge_buff=edge_buff, node_buff=node_buff, infill=infill)
    geo_s = gpd.GeoSeries(isochrone_polys)
    geo_df = gpd.GeoDataFrame(geometry=geo_s)
    geo_df = geo_df.set_crs(ox.graph_to_gdfs(G)[0].crs)
    return geo_df
