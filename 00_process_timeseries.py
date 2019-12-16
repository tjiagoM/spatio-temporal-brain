import numpy as np

from utils import NETMATS_PEOPLE, NEW_STRUCT_PEOPLE

filtered_people = sorted(list(set(NETMATS_PEOPLE) \
                              .intersection(set(NEW_STRUCT_PEOPLE))))


def get_bn_path(person, session):
    return f'../hcp_multimodal_parcellation/timeseries/{person}_{session}/{person}_rfMRI_REST{session}_rfMRI_REST{session}_hp2000_clean_BN_Atlas_246_2mm.txt'


def get_aal_path(person, session):
    return f'../hcp_multimodal_parcellation/timeseries/{person}_{session}/{person}_rfMRI_REST{session}_rfMRI_REST{session}_hp2000_clean_AAL3.txt'


def get_final_path(person, session_day):
    return f'../hcp_multimodal_parcellation/concatenated_timeseries/{person}_{session_day}.npy'


for person in filtered_people:
    for day_session in ['1', '2']:
        try:
            arr_bn = np.genfromtxt(get_bn_path(person, day_session + '_LR'))
            arr_aal = np.genfromtxt(get_aal_path(person, day_session + '_LR'))
            arr_concat_lr = np.concatenate([arr_bn, arr_aal[94:120]], axis=0)

            arr_bn = np.genfromtxt(get_bn_path(person, day_session + '_RL'))
            arr_aal = np.genfromtxt(get_aal_path(person, day_session + '_RL'))
            arr_concat_rl = np.concatenate([arr_bn, arr_aal[94:120]], axis=0)

            final_ts = np.concatenate([arr_concat_lr, arr_concat_rl], axis=1)

            final_path = get_final_path(person, day_session)
            # os.makedirs(os.path.dirname(final_path), exist_ok=True)
            np.save(final_path, final_ts)

        except Exception as e:
            print(person, day_session)
