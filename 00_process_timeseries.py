import numpy as np

from utils import NEW_MULTIMODAL_TIMESERIES, NEW_STRUCT_PEOPLE

filtered_people = sorted(list(set(NEW_MULTIMODAL_TIMESERIES)
                              .intersection(set(NEW_STRUCT_PEOPLE))))


def get_bn_path(person, session):
    return f'../hcp_multimodal_parcellation/timeseries/{person}_{session}/{person}_rfMRI_REST{session}_rfMRI_REST{session}_hp2000_clean_BN_Atlas_246_2mm.txt'


def get_aal_path(person, session):
    return f'../hcp_multimodal_parcellation/timeseries/{person}_{session}/{person}_rfMRI_REST{session}_rfMRI_REST{session}_hp2000_clean_AAL3.txt'


def get_final_path(person, session_day):
    return f'../hcp_multimodal_parcellation/concatenated_timeseries/{person}_{session_day}.npy'


# Just in case is needed
def check_timeseries():
    for person in filtered_people:
        for day_session in ['1_LR', '1_RL', '2_LR', '2_RL']:
            try:
                _ = np.genfromtxt(get_bn_path(person, day_session))
            except Exception:
                print(person, day_session)


for person in filtered_people:
    for day_session in ['1', '2']:
        try:
            arr_bn_lr = np.genfromtxt(get_bn_path(person, day_session + '_LR'))
            arr_aal_lr = np.genfromtxt(get_aal_path(person, day_session + '_LR'))
            arr_concat_lr = np.concatenate([arr_bn_lr, arr_aal_lr[94:120]], axis=0)

            arr_bn_rl = np.genfromtxt(get_bn_path(person, day_session + '_RL'))
            arr_aal_rl = np.genfromtxt(get_aal_path(person, day_session + '_RL'))
            arr_concat_rl = np.concatenate([arr_bn_rl, arr_aal_rl[94:120]], axis=0)

            final_ts = np.concatenate([arr_concat_lr, arr_concat_rl], axis=1)

            # Some timeseries are strangely truncated
            if final_ts.shape[0] != 272 or final_ts.shape[1] != 2400:
                print(person, day_session, final_ts.shape, arr_bn_lr.shape, arr_aal_lr.shape, arr_bn_rl.shape,
                      arr_aal_rl.shape)
                continue

            final_path = get_final_path(person, day_session)
            # os.makedirs(os.path.dirname(final_path), exist_ok=True)
            np.save(final_path, final_ts)

        except OSError:
            # This person+session does not have timeseries available
            print(person, day_session)
