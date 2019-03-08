import pandas as pd
from glob import glob
from tqdm import tqdm
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import numpy as np


movers = dd.read_csv(
    '/home/data/infutor/CRD4/bay_area_movers*.csv',
    # dtype={'county_seq_' + str(x): str for x in range(1, 11)},
    dtype=str,
    assume_missing=True
)

bay_area_counties = [
    '001', '013', '041', '055', '075', '081', '085', '095', '097']

all_moves = pd.DataFrame()
tossed_pids = []

for i, row in tqdm(movers.iterrows(), total=len(movers)):

    valid = True
    tmp = pd.DataFrame(columns=[
        'pid', 'from_addrid', 'to_addrid', 'from_effdate',
        'to_effdate', 'from_county', 'to_county'])

    for x in range(1, 10):

        from_county_col = 'county_seq_' + str(x)
        to_county_col = 'county_seq_' + str(x + 1)
        from_addrid_col = 'addrid_seq_' + str(x)
        to_addrid_col = 'addrid_seq_' + str(x + 1)
        from_effdate_col = 'effdate_seq_' + str(x)
        to_effdate_col = 'effdate_seq_' + str(x + 1)

        if (pd.notnull(row[from_effdate_col])) & \
                (pd.notnull(row[to_effdate_col])):
            if (row[from_county_col] in bay_area_counties) & \
                    (row[to_county_col] in bay_area_counties):
                if row[from_addrid_col] != row[to_addrid_col]:
                    row_df = pd.DataFrame([row[[
                        'pid_a', from_addrid_col, to_addrid_col,
                        from_effdate_col, to_effdate_col,
                        from_county_col, to_county_col,
                    ]]]).rename(columns={
                        'pid_a': 'pid', from_addrid_col: 'from_addrid',
                        to_addrid_col: 'to_addrid',
                        from_effdate_col: 'from_effdate',
                        to_effdate_col: 'to_effdate',
                        from_county_col: 'from_county',
                        to_county_col: 'to_county'
                    })
                    tmp = pd.concat([tmp, row_df], axis=0, ignore_index=True)

        elif ((pd.isnull(row[from_effdate_col])) &
                (pd.notnull(row[from_addrid_col]))):
            valid = False
            tossed_pids.append(row['pid_a'])
            break

        elif ((pd.isnull(row[to_effdate_col])) &
                (pd.notnull(row[to_addrid_col]))):
            valid = False
            tossed_pids.append(row['pid_a'])
            break

    if valid:
        all_moves = pd.concat((all_moves, tmp))

print(tossed_pids)
print(len(tossed_pids))
all_moves.to_csv('../all_moves.csv', index=False)
