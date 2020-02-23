import os

import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure

from utils import UKB_IDS_PATH, UKB_ADJ_ARR_PATH, UKB_TIMESERIES_PATH, UKB_PHENOTYPE_PATH

IDS_TO_EXCLUDE = ['UKB2203847_ts_raw.txt', 'UKB2208238_ts_raw.txt', 'UKB2697888_ts_raw.txt']

conn_measure = ConnectivityMeasure(
    kind='correlation',
    vectorize=False)  # return will be in (376x376) shape instead of 1D vector

if __name__ == '__main__':
    # Getting all UK IDs and save it for future use, but only for those who we have IDs
    timeseries_ids = [int(f[len('UKB'):-len("_ts_raw.txt")]) for f in os.listdir(UKB_TIMESERIES_PATH) if f not in IDS_TO_EXCLUDE]
    info_df = pd.read_csv(UKB_PHENOTYPE_PATH, delimiter=',').set_index('eid')['31-0.0']
    phenotype_ids = info_df.index.values
    timeseries_ids = sorted(set(timeseries_ids).intersection(set(phenotype_ids)))

    np.save(UKB_IDS_PATH, timeseries_ids)

    for person in timeseries_ids:
        ts = np.loadtxt(f'{UKB_TIMESERIES_PATH}/UKB{person}_ts_raw.txt', delimiter=',')
        if ts.shape[1] == 523:
            print('.', end='')
            ts = ts[:, :490]

        assert ts.shape == (376, 490)

        corr_mat = conn_measure.fit_transform([ts.T])
        assert corr_mat.shape == (1, 376, 376)

        np.save(f'{UKB_ADJ_ARR_PATH}/{person}.npy', corr_mat[0])
