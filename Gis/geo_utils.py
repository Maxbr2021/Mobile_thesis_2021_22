import geopandas as gpd
import pandas as pd
import numpy as np
import datetime
import shapely
import osmnx as ox


### convert df to gdf
def gdf_from_df(df,lat_lon_col_names):
    lat,lon = lat_lon_col_names[0], lat_lon_col_names[1]
    gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df[f'{lon}'],df[f'{lat}'])).drop(lat_lon_col_names,axis=1)
    gdf["tracked_at_date"] = pd.to_datetime(gdf["tracked_at_date"])
    gdf = gdf.set_index('tracked_at_date')
    gdf['tracked_at_date'] = gdf.index
    gdf.index = gdf.index.date
    gdf = gdf.sort_values(by=['tracked_at_date'])
    return gdf

### get dates and convex hull for each day
def create_convex_hulls(gdf):
    dates = np.unique(np.array(gdf.index))
    #### create empty gdf #####
    convex_hulls_gdf = gpd.GeoDataFrame()
    #### filter by day and add convex hull per day
    convex_hulls = []
    for day in dates:
        try:
            d = gdf.loc[day,:]
            convex_hulls.append(d.unary_union.convex_hull)
        except:
            print(f"{d} at {day} just one sample")
            d = gdf.loc[[day]]
            convex_hulls.append(d.unary_union.convex_hull)
    ##### save convex hulls in gdf
    convex_hulls_gdf['geometry'] = convex_hulls
    convex_hulls_gdf = convex_hulls_gdf.set_crs('epsg:4326') #set to wsg84 (lon,lat)
    convex_hulls_gdf = convex_hulls_gdf.set_index(dates) #set dates as index again
    return convex_hulls_gdf,dates


### calculate ptc of overlapping convex hull per day 
def RevisitedLS(convex_hulls_gdf,dates,proj=None):
    gdf_p = ox.project_gdf(convex_hulls_gdf,to_crs=proj)
    gdf_p = gdf_p.set_index(dates)
    ptc = []
    ### intersection of day x with all other days ####
    for day in dates:
        day_of_interest = gdf_p.loc[[day]]
        other_days = gdf_p.loc[gdf_p.index != day, :]
        inter = gpd.overlay(day_of_interest, other_days, how='intersection')
        # inter = inter.set_index(dates[dates != day])
        if len(inter) != 0:
            ptc_day = inter.unary_union.area / day_of_interest.unary_union.area
        else:
            ptc_day = 0
        ptc.append(ptc_day)
    s = pd.Series(ptc,index=dates)
    return round(s * 100,3)

### check if convex_hull plygon is a valid polygon ... else remove it
def check_samples(df,date_df):
    if len(df[df.geometry.geom_type == 'Point']) > 0 or len(df[df.geometry.geom_type == 'LineString']) > 0:
        if len(df[df.geometry.geom_type == 'Point']) > 0:
            i = df[df.geometry.geom_type == 'Point'].index
            df = df.drop(i)
            date_df = date_df.drop(i)
            new_dates = np.unique(np.array(df.index))
        if len(df[df.geometry.geom_type == 'LineString']) > 0:
            i = df[df.geometry.geom_type == 'LineString'].index
            df = df.drop(i)
            date_df = date_df.drop(i)
            new_dates = np.unique(np.array(df.index))
        return df,date_df,new_dates
    else:
        return df,date_df,np.unique(np.array(df.index))

def CHull_skm(convex_hulls_gdf,proj=None):
    gdf_p = ox.project_gdf(convex_hulls_gdf,to_crs=proj)
    return gdf_p.unary_union.convex_hull.area / 1e6

def CHull_skm_daily(convex_hulls_gdf,proj=None):
    gdf_p = ox.project_gdf(convex_hulls_gdf,to_crs=proj)
    return gdf_p.area / 1e6

### return compactness of lifespace 
def GravCompact(convex_hulls_gdf,proj=None):
    gdf_p = ox.project_gdf(convex_hulls_gdf,to_crs=proj)
    A = gdf_p.unary_union.convex_hull.area
    P = gdf_p.unary_union.convex_hull.length
    return P / (2*np.sqrt(np.pi*A))

### return compactness of lifespace per day 
def GravCompact_daily(convex_hulls_gdf,proj=None):
    gdf_p = ox.project_gdf(convex_hulls_gdf,to_crs=proj)
    A = gdf_p.area
    P = gdf_p.length
    return P / (2*np.sqrt(np.pi*A))

def day_convex_overlapping(convex_hulls_gdf,dates, proj=None):
    gdf_p = ox.project_gdf(convex_hulls_gdf,to_crs=proj)
    gdf_p = gdf_p.set_index(dates)
    ## build gdf form projected gdf union convex hull geometry
    overall_convex_hull = gpd.GeoSeries(gdf_p.unary_union.convex_hull)
    overall_convex_hull_df = gpd.GeoDataFrame(geometry=overall_convex_hull).set_crs(gdf_p.crs)
    ptc = []
    for day in dates:
        inter = gpd.overlay(gdf_p.loc[[day]], overall_convex_hull_df, how='intersection',keep_geom_type=False)
        if len(inter) != 0:
            ptc.append(inter.unary_union.area / overall_convex_hull_df.area[0])
        else:
            ptc.append(0)
    s = pd.Series(ptc,index=dates)
    return round(s * 100,2)

## helper for overlay combine gdf to unary_union polygon and return as gdf
def polygon_to_gdf(gdf_p,proj):
    polygon = gdf_p.unary_union
    s = gpd.GeoSeries(polygon)
    gdf = gpd.GeoDataFrame(geometry=s).set_crs(proj)
    return gdf

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

def create_point_buffer(lon_lat,dist,proj=None):
    point = build_geo_df(lon_lat)
    point_proj = ox.project_gdf(point,to_crs=proj)
    buffer = gpd.GeoDataFrame(geometry=point_proj.buffer(dist)).set_crs(point_proj.crs)
    return buffer
