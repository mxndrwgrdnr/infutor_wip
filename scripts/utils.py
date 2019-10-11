import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from shapely.affinity import scale
from matplotlib.patches import Arc
import dask.dataframe as dd


def get_mover_counts(
        movers_file, st_mo=190001, end_mo=204001, demog_query_str=None):
    movers = dd.read_csv(movers_file, dtype={
        'PROP_FIPSCD_from': str, 'PROP_FIPSCD_to': str,
        'PROP_CENSUSTRACT_to': str, 'PROP_CENSUSTRACT_from': str},
        blocksize=25e6
    ).rename(columns={'Unnamed: 0': 'pid'}).set_index('pid')
    movers = movers[
        (movers['to_effdate'] > st_mo) & (movers['to_effdate'] < end_mo)]

    if demog_query_str:
        demog_cols = [
            'PID', 'AGE', 'LOR', 'HOMEOWNERCD', 'EHI', 'PCTB', 'PCTW',
            'PCTA', 'PCTH']
        demog_dtypes = {
            col: float if col in ['AGE', 'LOR'] else str for col in demog_cols}
        demog = dd.read_csv(
            '/home/data/infutor/NARC3/bay_area_mover*.csv',
            usecols=demog_cols,
            assume_missing=True,
            dtype=demog_dtypes).set_index('PID')
        demog = demog.query(demog_query_str)
        movers_w_demog = movers.merge(demog, left_index=True, right_index=True)
        movers = movers_w_demog.compute()
    movers.index.name = 'PID'
    movers = movers.reset_index()
    grouped = movers[['PID', 'PROP_FIPSCD_from', 'PROP_FIPSCD_to']].groupby(
        ['PROP_FIPSCD_from', 'PROP_FIPSCD_to'])
    flow_counts = grouped.count().reset_index()
    flow_counts.rename(columns={
        'PID': 'count', 'PROP_FIPSCD_to': 'FIPSCD_to',
        'PROP_FIPSCD_from': 'FIPSCD_from'}, inplace=True)
    total_infutor_movers = flow_counts['count'].sum()
    flow_counts['pct_flow'] = flow_counts['count'] / total_infutor_movers
    return flow_counts


def _make_circular_arrow(centroid_x, centroid_y, lw, ax):
    x0, y0 = centroid_x, centroid_y
    radius = 0.04
    for angle in [120, 300]:
        angle_rad = angle * np.pi / 180  # degrees to radians
        arc = Arc((x0, y0),
                  radius * 2, radius * 2,  # ellipse width and height
                  theta1=angle - 120, theta2=angle,
                  linewidth=lw)
        ax.add_patch(arc)
        arc_arrow_length = 0.002
        arc_arrow_dx = arc_arrow_length * np.cos(angle_rad + np.pi / 2)
        arc_arrow_dy = arc_arrow_length * np.sin(angle_rad + np.pi / 2)
        ax.arrow(
            x0 + radius * np.cos(angle_rad) - arc_arrow_dx,
            y0 + radius * np.sin(angle_rad) - arc_arrow_dy,
            # We want to define a vector,
            # but we don't want to draw any line besides arrow head,
            # so we make arrow "body" unnoticeable.
            arc_arrow_dx * 0.000001,
            arc_arrow_dy * 0.000001,
            head_width=lw / 100,
            head_length=lw / 150,
            color='black')


def proportional_scale(linestring, min_length, scale_factor):

    loss = min_length - min_length * scale_factor
    proportional_factor = 1 - loss / linestring.length
    scaled_linestring = scale(
        linestring, proportional_factor, proportional_factor)
    return scaled_linestring


def plot_flows(
        ax, polygon_gdf, polygon_key, flow_data, flow_value_col,
        flow_key_from, flow_key_to, kind='directional', legend_scale='linear',
        normalized=False, show_axes=False):

    centroids = polygon_gdf.copy()
    centroids['geometry'] = centroids.centroid

    if kind != 'net':

        count_divisor = 11000
        legend_levels = [1000, 5000, 10000, 20000, 50000]
        legend_title = 'total movers'
        scale_factor = 0.4

        if kind == 'directional':
            style = 'arrows'

        elif kind == 'total':
            style = 'lines'
            total_counts = flow_data.copy()
            total_counts['sorted_from'] = None
            total_counts['sorted_to'] = None
            total_counts[['sorted_from', 'sorted_to']] = np.sort(
                total_counts[[flow_key_from, flow_key_to]], 1)
            total_counts_cols = ['sorted_from', 'sorted_to', flow_value_col]
            total_counts = total_counts[total_counts_cols].groupby(
                ['sorted_from', 'sorted_to']).sum().reset_index()
            flow_key_from = 'sorted_from'
            flow_key_to = 'sorted_to'
            flow_data = total_counts

    elif kind == 'net':
        style = 'arrows'
        scale_factor = 0.6
        count_divisor = 500
        legend_levels = [100, 500, 1000, 2500, 5000]
        legend_title = 'net movers'
        net_counts = flow_data.copy()
        net_counts['sorted_from'] = None
        net_counts['sorted_to'] = None
        net_counts[['sorted_from', 'sorted_to']] = np.sort(
            net_counts[[flow_key_from, flow_key_to]], 1)
        net_counts['net_count'] = net_counts[flow_value_col]
        reverse_flow_mask = net_counts[
            flow_key_from] != net_counts['sorted_from']
        net_counts.loc[reverse_flow_mask, 'net_count'] = net_counts.loc[
            reverse_flow_mask, 'net_count'] * -1
        net_counts_cols = ['sorted_from', 'sorted_to', 'net_count']
        net_counts = net_counts[net_counts_cols].groupby(
            ['sorted_from', 'sorted_to']).sum().reset_index()
        net_counts['net_from'] = net_counts['sorted_from']
        net_counts['net_to'] = net_counts['sorted_to']
        neg_flow_mask = net_counts['net_count'] < 0
        net_counts.loc[neg_flow_mask, 'net_from'] = net_counts.loc[
            neg_flow_mask, 'sorted_to']
        net_counts.loc[neg_flow_mask, 'net_to'] = net_counts.loc[
            neg_flow_mask, 'sorted_from']
        net_counts = net_counts[['net_from', 'net_to', 'net_count']]
        net_counts[flow_value_col] = np.abs(net_counts['net_count'])
        flow_key_from = 'net_from'
        flow_key_to = 'net_to'
        flow_data = net_counts

    flow_geog = pd.merge(
        flow_data, centroids[[polygon_key, 'geometry']],
        left_on=flow_key_from, right_on=polygon_key)
    flow_geog.rename(columns={'geometry': 'centroid_from'}, inplace=True)
    flow_geog.drop(columns=[polygon_key], inplace=True)

    flow_geog = pd.merge(
        flow_geog, centroids[[polygon_key, 'geometry']],
        left_on=flow_key_to, right_on=polygon_key)
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
    flow_geog.crs = {'init': 'epsg:4326'}
    min_geom_length = flow_geog['geometry'].length.min()

    flow_geog['geometry'] = flow_geog['geometry'].apply(
        proportional_scale, args=(min_geom_length, scale_factor))

    intra_mask = flow_geog[flow_key_from] == flow_geog[flow_key_to]

    polygon_gdf.plot(ax=ax, facecolor='None', edgecolor='grey')

    # return flow_geog
    if flow_value_col == 'pct_flow':

        if kind != 'net':
            count_divisor = 0.03
            legend_levels = [0.01, 0.02, 0.05, 0.1, 0.2]
        else:
            count_divisor = 0.0015
            legend_levels = [0.001, 0.005, 0.01, 0.02]

    if style == 'lines':
        flow_geog[~intra_mask].plot(
            linewidth=flow_geog.loc[
                ~intra_mask, flow_value_col] / count_divisor,
            ax=ax, edgecolor='k')
        for i, row in flow_geog[intra_mask].iterrows():
            _make_circular_arrow(
                *list(row['centroid_to'].coords)[0],
                lw=row[flow_value_col] / count_divisor, ax=ax)

    elif style == 'arrows':
        for i, row in flow_geog.iterrows():
            if row[flow_key_from] != row[flow_key_to]:
                x1, y1 = row['geometry'].coords[0]
                x2, y2 = row['geometry'].coords[1]
                ax.arrow(
                    x1, y1, x2 - x1, y2 - y1,
                    linewidth=row[flow_value_col] / count_divisor,
                    head_length=0.02,
                    head_width=0.02,
                    facecolor='k',
                    length_includes_head=True
                )

        if kind != 'net':
            intra_flows = flow_geog[intra_mask]

            for i, row in intra_flows.iterrows():
                _make_circular_arrow(
                    *list(row['centroid_to'].coords)[0],
                    lw=row[flow_value_col] / count_divisor, ax=ax)

    for i in legend_levels:
        if flow_value_col == 'pct_flow':
            ax.plot(
                np.NaN, np.NaN, '-', color='k', label=str(i * 100) + '%',
                linewidth=i / count_divisor)
        else:
            ax.plot(
                np.NaN, np.NaN, '-', color='k', label=str(i),
                linewidth=i / count_divisor)

    ax.legend(title=legend_title, fontsize=13, borderpad=1, title_fontsize=13)

    if not show_axes:
        ax.axis('off')
