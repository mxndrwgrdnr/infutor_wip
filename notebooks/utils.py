import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from shapely.affinity import scale

def plot_flows(ax, polygon_gdf, polygon_key, flow_data, flow_value_col, flow_key_from, flow_key_to, kind='directional'):
    
    centroids = polygon_gdf.copy()
    centroids['geometry'] = centroids.centroid
    
    def proportional_scale(linestring, min_length, scale_factor):
    
        loss = min_length - min_length * scale_factor
        proportional_factor = 1 - loss/linestring.length
        scaled_linestring = scale(linestring, proportional_factor, proportional_factor)
        return scaled_linestring
    
    if kind == 'directional':
        style = 'arrows'
    
    if kind == 'total':
        style = 'lines'
        total_counts = flow_data.copy()
        total_counts['sorted_from'] = None
        total_counts['sorted_to'] = None
        total_counts[['sorted_from', 'sorted_to']] = np.sort(total_counts[[flow_key_from, flow_key_to]], 1)
        total_counts = total_counts[['sorted_from', 'sorted_to', flow_value_col]].groupby(
            ['sorted_from', 'sorted_to']).sum().reset_index()
        flow_key_from = 'sorted_from'
        flow_key_to = 'sorted_to'
        flow_data = total_counts
    
    elif kind == 'net':
        style = 'arrows'
        net_counts = flow_data.copy()
        net_counts['sorted_from'] = None
        net_counts['sorted_to'] = None
        net_counts[['sorted_from', 'sorted_to']] = np.sort(net_counts[[flow_key_from, flow_key_to]], 1)
        net_counts['net_count'] = net_counts[flow_value_col]
        reverse_flow_mask = net_counts[flow_key_from] != net_counts['sorted_from']
        net_counts.loc[reverse_flow_mask, 'net_count'] = net_counts.loc[reverse_flow_mask, 'net_count'] * -1
        net_counts = net_counts[['sorted_from', 'sorted_to', 'net_count']].groupby(
            ['sorted_from', 'sorted_to']).sum().reset_index()
        net_counts['net_from'] = net_counts['sorted_from']
        net_counts['net_to'] = net_counts['sorted_to']
        net_counts.loc[net_counts['net_count'] < 0, 'net_from'] = net_counts.loc[net_counts['net_count'] < 0, 'sorted_to']
        net_counts.loc[net_counts['net_count'] < 0, 'net_to'] = net_counts.loc[net_counts['net_count'] < 0, 'sorted_from']
        net_counts = net_counts[['net_from', 'net_to', 'net_count']]
        net_counts[flow_value_col] = np.abs(net_counts['net_count'])
        flow_key_from = 'net_from'
        flow_key_to = 'net_to'
        flow_data = net_counts
    
    flow_geog = pd.merge(flow_data, centroids[[polygon_key, 'geometry']], left_on=flow_key_from, right_on=polygon_key)
    flow_geog.rename(columns={'geometry': 'centroid_from'}, inplace=True)
    flow_geog.drop(columns=[polygon_key], inplace=True)

    flow_geog = pd.merge(flow_geog, centroids[[polygon_key, 'geometry']], left_on=flow_key_to, right_on=polygon_key)
    flow_geog.rename(columns={'geometry': 'centroid_to'}, inplace=True)
    flow_geog.drop(columns=[polygon_key], inplace=True)

    geoms = []
    for i, row in flow_geog.iterrows():
        if row[flow_key_from] != row[flow_key_to]:
            line = LineString((row['centroid_from'], row['centroid_to']))
            if kind == 'directional':
                line = line.parallel_offset(0.025, side='right')
            geoms.append(line)
        else:
            geoms.append(row['centroid_to'].buffer(0.04).boundary)

    flow_geog['geometry'] = geoms
    flow_geog = gpd.GeoDataFrame(flow_geog, geometry='geometry')
    flow_geog.crs = {'init' :'epsg:4326'}
    min_geom_length = flow_geog['geometry'].length.min()
    flow_geog['geometry'] = flow_geog['geometry'].apply(proportional_scale, args=(min_geom_length, 0.5))
    
    polygon_gdf.plot(ax=ax, facecolor='None', edgecolor='k', label='doo')
    
    if kind == 'net':
        count_divisor = 500
        legend_levels = [100, 500, 1000, 2500, 5000]
    else:
        count_divisor = 5000
        legend_levels = [250, 1000, 5000, 10000, 20000, 50000]
        
    if style == 'lines':
        flow_geog.plot(linewidth=flow_geog[flow_value_col]/count_divisor, ax=ax, edgecolor='k')
        
    elif style == 'arrows':
        
        for i, row in flow_geog.iterrows():
            if row[flow_key_from] != row[flow_key_to]:
                ax.arrow(
                    row['geometry'].coords[0][0],
                    row['geometry'].coords[0][1],
                    row['geometry'].coords[1][0] - row['geometry'].coords[0][0],
                    row['geometry'].coords[1][1] - row['geometry'].coords[0][1],
                    linewidth=row[flow_value_col]/count_divisor,
                    head_length=0.03,
                    head_width=0.02,
                    facecolor='k',
                    length_includes_head=True
                )
        if kind != 'net':
            intra_flows = flow_geog[flow_geog[flow_key_from] == flow_geog[flow_key_to]]
            intra_flows.plot(linewidth=intra_flows[flow_value_col]/count_divisor, ax=ax, edgecolor='k')
    for i in legend_levels:
        ax.plot(np.NaN, np.NaN, '-', color='k', label=str(i), linewidth=i/count_divisor)
    
    if kind != 'net':
        ax.legend(title='total movers', fontsize=13, borderpad=1, title_fontsize=13)
    else:
        ax.legend(title='net movers', fontsize=13, borderpad=1, title_fontsize=13)