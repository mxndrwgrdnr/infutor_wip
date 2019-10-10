import pandas as pd
from glob import glob
from tqdm import tqdm
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.multiprocessing import get
import numpy as np
import pyarrow
from shapely.geometry import Point
import geopandas as gpd


def process_df(df):

    out_cols = [
        'pid', 'from_addrid', 'to_addrid', 'from_effdate',
        'to_effdate', 'from_county', 'to_county', 'seq']
    long_moves = pd.DataFrame(columns=out_cols, dtype=str)

    for x in range(1, 10):

        from_county_col = 'county_seq_' + str(x)
        to_county_col = 'county_seq_' + str(x + 1)
        from_addrid_col = 'addrid_seq_' + str(x)
        to_addrid_col = 'addrid_seq_' + str(x + 1)
        from_effdate_col = 'effdate_seq_' + str(x)
        to_effdate_col = 'effdate_seq_' + str(x + 1)

        tmp = df[[
            'pid_a', from_addrid_col, to_addrid_col, from_effdate_col,
            to_effdate_col, from_county_col, to_county_col]].copy(deep=True)
        tmp.loc[:, 'seq'] = x
        long_moves = pd.concat((
            long_moves, tmp.rename(columns=dict(zip(tmp.columns, out_cols)))))

    return long_moves


def get_dist(df):

    moves_w_geog = df.copy()
    moves_w_geog['from_coords'] = list(zip(
        moves_w_geog.from_lon, moves_w_geog.from_lat))
    moves_w_geog['to_coords'] = list(zip(
        moves_w_geog.to_lon, moves_w_geog.to_lat))
    moves_w_geog['from_coords'] = moves_w_geog['from_coords'].apply(Point)
    moves_w_geog['to_coords'] = moves_w_geog['to_coords'].apply(Point)
    from_gs = gpd.GeoSeries(
        moves_w_geog['from_coords'],
        crs={'init': 'epsg:4326'}).to_crs(epsg='2768')
    to_gs = gpd.GeoSeries(
        moves_w_geog['to_coords'],
        crs={'init': 'epsg:4326'}).to_crs(epsg='2768')
    dists = from_gs.distance(to_gs)
    moves_w_geog['distance'] = dists

    return moves_w_geog[np.append(df.columns, 'distance')]


out_cols = [
    'pid', 'from_addrid', 'to_addrid', 'from_effdate', 'to_effdate',
    'from_county', 'to_county', 'seq']
demog_cols = [
    'PID', 'AGE', 'LOR', 'HOMEOWNERCD', 'EHI', 'PCTB', 'PCTW', 'PCTA', 'PCTH']
demog_dtypes = {
    col: float if col in ['AGE', 'LOR'] else str for col in demog_cols}


if __name__ == '__main__':

    # clean up movers data
    movers = dd.read_csv(
        '../data/bay_area_movers*.csv',
        dtype={'county_seq_' + str(x): str for x in range(1, 11)},
        assume_missing=True
    )

    movers['not_valid'] = (
        (movers.addrid_seq_1.notnull() & movers.effdate_seq_1.isna()) |
        (movers.addrid_seq_2.notnull() & movers.effdate_seq_2.isna()) |
        (movers.addrid_seq_3.notnull() & movers.effdate_seq_3.isna()) |
        (movers.addrid_seq_4.notnull() & movers.effdate_seq_4.isna()) |
        (movers.addrid_seq_5.notnull() & movers.effdate_seq_5.isna()) |
        (movers.addrid_seq_6.notnull() & movers.effdate_seq_6.isna()) |
        (movers.addrid_seq_7.notnull() & movers.effdate_seq_7.isna()) |
        (movers.addrid_seq_8.notnull() & movers.effdate_seq_8.isna()) |
        (movers.addrid_seq_9.notnull() & movers.effdate_seq_9.isna()) |
        (movers.addrid_seq_10.notnull() & movers.effdate_seq_10.isna()))

    with ProgressBar():
        validated_movers = movers.compute()

    cleaned_movers = validated_movers[validated_movers['not_valid'] is False]
    cleaned_movers.to_parquet(
        '../data/cleaned_movers.parquet', engine='pyarrow')

    # convert movers wide to long
    cleaned_movers = dd.read_parquet(
        '../data/cleaned_movers.parquet', engine='pyarrow')
    cleaned_movers = cleaned_movers.repartition(npartitions=10000)
    long_movers = cleaned_movers.map_partitions(
        process_df, meta=pd.DataFrame(columns=out_cols, dtype=str))

    with ProgressBar():
        all_moves = long_movers.compute()

    assert len(all_moves) == len(cleaned_movers) * 9

    # Drop rows without full to/from data
    all_moves = dd.from_pandas(all_moves, npartitions=10000)
    moves_not_null = all_moves[
        all_moves['from_effdate'].notnull() &
        all_moves['to_effdate'].notnull()]

    with ProgressBar():
        moves_not_null = moves_not_null.compute()

    print('{0} non-null moves'.format(len(moves_not_null)))

    # Drop rows where move is between the same address ID
    moves_not_dupe = moves_not_null[
        moves_not_null['from_addrid'] != moves_not_null['to_addrid']]
    print('{0} non-duplicate moves'.format(len(moves_not_dupe)))

    moves_not_dupe.to_parquet('../data/moves_long.parquet', engine='pyarrow')
    del movers, validated_movers, cleaned_movers, long_movers, \
        all_moves, moves_not_null, moves_not_dupe

    moves = dd.read_parquet(
        '../data/moves_long.parquet',
        columns=['pid', 'from_addrid', 'to_addrid', 'to_effdate'])
    num_moves = len(moves)

    # Clean up properties data
    properties = dd.read_csv(
        '../data/bay_area_properties*',
        dtype={'PROP_CENSUSTRACT': str, 'PROP_FIPSCD': str},
        usecols=[
            'ADDRID', 'ADDRID2', 'DPV', 'PROP_LATITUDE', 'PROP_LONGITUDE',
            'PROP_FIPSCD', 'PROP_CENSUSTRACT', 'PROP_MUNINAME',
            'PROP_OWNEROCC', 'PROP_QLTY', 'PROP_VALCALC', 'PROP_UNVBLDSQFT',
            'PROP_BEDRMS']
    )

    prop_w_geog = properties[
        properties['PROP_LATITUDE'].notnull() &
        properties['PROP_LONGITUDE'].notnull() &
        properties['DPV'].isin(['D', 'S', 'Y']) &
        properties['PROP_FIPSCD'].isin([
            '06001', '06013', '06041', '06055', '06075', '06081', '06085',
            '06095', '06097'])
    ]

    # Merge moves with "from" properties
    moves = moves.set_index('from_addrid')
    prop_w_geog = prop_w_geog.set_index('ADDRID')
    merged_1 = moves.merge(prop_w_geog, left_index=True, right_index=True)

    with ProgressBar():
        merged_1 = merged_1.rename(columns={
            'PROP_LATITUDE': 'from_lat',
            'PROP_LONGITUDE': 'from_lon'}).compute()

    moves_w_from_geog = merged_1.reset_index().rename(
        columns={'index': 'from_addrid'})
    num_matched_on_from = len(moves_w_from_geog)
    pct_matched = round(num_matched_on_from / num_moves, 1) * 100
    print('Matched {0} of {1} ({2}%) moves on "to" address'.format(
        num_matched_on_from, num_moves, pct_matched))

    # Merge moves with "to" properties
    moves_w_from_geog = moves_w_from_geog.set_index('to_addrid')
    merged_2 = prop_w_geog.merge(
        moves_w_from_geog, left_index=True, right_index=True,
        suffixes=('_to', '_from'))

    with ProgressBar():
        merged_2 = merged_2.rename(columns={
            'PROP_LATITUDE': 'to_lat', 'PROP_LONGITUDE': 'to_lon'}).compute()

    moves_w_geog = merged_2.reset_index().rename(
        columns={'index': 'to_addrid'})
    num_matched_on_to = len(moves_w_geog)
    pct_matched = round(num_matched_on_to / num_matched_on_from, 1) * 100
    print('Matched {0} of {1} ({2}%) moves on "to" address'.format(
        num_matched_on_to, num_matched_on_from, pct_matched))

    # Drop duplicates
    uniq_moves_w_geog = moves_w_geog.drop_duplicates(
        ['pid', 'from_addrid', 'to_addrid'])

    num_matched_moves = len(uniq_moves_w_geog)
    print(
        'Matched both properties to property records with geographies '
        'for {0} of {1} ({2}%) of relocation records.'.format(
            str(num_matched_moves), str(num_moves),
            str(round(100 * num_matched_moves / num_moves, 1))))

    # Create geography columns
    uniq_moves_w_geog2 = uniq_moves_w_geog[[
        'pid', 'from_addrid', 'from_lat', 'from_lon', 'PROP_FIPSCD_from',
        'PROP_CENSUSTRACT_from', 'PROP_MUNINAME_from', 'PROP_OWNEROCC_from',
        'PROP_QLTY_from', 'PROP_VALCALC_from', 'PROP_UNVBLDSQFT_from',
        'PROP_BEDRMS_from', 'to_addrid', 'to_lat', 'to_lon', 'PROP_FIPSCD_to',
        'PROP_CENSUSTRACT_to', 'PROP_MUNINAME_to', 'PROP_OWNEROCC_to',
        'PROP_QLTY_to', 'PROP_VALCALC_to', 'PROP_UNVBLDSQFT_to',
        'PROP_BEDRMS_to', 'to_effdate'
    ]]

    ddf = dd.from_pandas(uniq_moves_w_geog2, npartitions=10000)
    dtypes = uniq_moves_w_geog2.dtypes.apply(lambda x: x.name).to_dict()
    dtypes.update({'distance': 'float'})
    ddf = ddf.map_partitions(get_dist, meta=dtypes)

    with ProgressBar():
        moves_w_dists = ddf.compute()

    moves_w_dists.to_csv('../data/moves_w_dists.csv', index=False)

    del moves, properties, prop_w_geog, merged_1, merged_2, \
        moves_w_from_geog, moves_w_geog, uniq_moves_w_geog, \
        uniq_moves_w_geog2, ddf, moves_w_dists

    # Merge demographics
    moves = dd.read_csv(
        '../data/moves_w_dists.csv', dtype={
            'PROP_CENSUSTRACT_from': str, 'PROP_FIPSCD_from': str,
            'PROP_CENSUSTRACT_to': str, 'PROP_FIPSCD_to': str},
        blocksize=25e6).set_index('pid')

    demog = dd.read_csv(
        '../data/bay_area_mover_demogs*.csv',
        usecols=demog_cols,
        assume_missing=True,
        dtype=demog_dtypes).set_index('PID')

    moves_w_demog = moves.merge(demog, left_index=True, right_index=True)

    with ProgressBar():
        moves_w_demog = moves_w_demog.compute()

    num_total_moves = len(moves)
    num_moves_w_demog = len(moves_w_demog)

    print(
        'Matched movers to demographic profiles '
        'for {0} of {1} ({2}%) of relocation records.'.format(
            str(num_moves_w_demog), str(num_total_moves), str(round(
                100 * num_moves_w_demog / num_total_moves, 1))))

    moves_w_demog.to_csv('../data/movers.csv')
