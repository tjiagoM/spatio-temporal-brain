from nilearn.connectome import ConnectivityMeasure
import numpy as np
import os

IDS_TO_EXCLUDE = ['UKB2203847_ts_raw.txt', 'UKB2208238_ts_raw.txt', 'UKB2697888_ts_raw.txt']
TIMESERIES_PATH = '../uk_biobank_dataset/ts_raw/'
SAVING_PATH = '../uk_biobank_dataset/cache_adj_arrs/'

conn_measure = ConnectivityMeasure(
    kind='correlation',
    vectorize=False)  # return will be in (376x376) shape instead of 1D vector

if __name__ == '__main__':
    # Getting all UK IDs and save it for future use
    timeseries_ids = [f[:-len("_ts_raw.txt")] for f in sorted(os.listdir(TIMESERIES_PATH)) if f not in IDS_TO_EXCLUDE]
    np.save(os.path.join('meta_data', 'ukb_ids.npy'), timeseries_ids)


    for person in timeseries_ids:
        ts = np.loadtxt(f'{TIMESERIES_PATH}{person}_ts_raw.txt', delimiter=',')
        if ts.shape[1] == 523:
            ts = ts[:,:490]

        assert ts.shape == (376, 490)

        corr_mat = conn_measure.fit_transform([ts.T])
        assert corr_mat.shape == (1, 376, 376)

        np.save(f'{SAVING_PATH}{person}.npy', corr_mat[0])
