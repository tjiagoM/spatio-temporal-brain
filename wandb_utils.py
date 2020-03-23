from enum import Enum, unique

@unique
class SweepType(str, Enum):
    DIFFPOOL = 'diff_pool'
    MESSAGE_PASSING = 'message_passing'

OPTIONS_DIFFPOOL = {
    'pooling' : {
        'value': 'diff_pool'
    },
    'threshold': {
        'distribution': 'categorical',
        'values': [5, 20]
    }
}

OPTIONS_MESSAGE_PASSING = {
    'pooling': {
        'distribution': 'categorical',
        'values': ['mean', 'concat']
    },
    'gnn_type': {
        'distribution': 'categorical',
        'values': ['none',
                   #
                   'GCN-1-5',
                   'GCN-1-20',
                   'GCN-2-5',
                   'GCN-2-20',
                   #
                   'GAT-1-1-5',
                   'GAT-1-1-20',
                   'GAT-2-1-5',
                   'GAT-2-1-20',
                   'GAT-1-5-5',
                   'GAT-1-5-20',
                   'GAT-2-5-5',
                   'GAT-2-5-20']
    }
}


SWEEP_GENERAL = {
    # name to be defined elsewhere
    'metric': {
        'goal': 'minimize',
        'name': 'best_val_loss'
    },
    'method': 'bayes',
    'parameters': {
        'activation': {
            'distribution': 'categorical',
            'values': ['relu']
        },
        'encoding_strategy': {
            'distribution': 'categorical',
            'values': ['none']
        },
        'conv_strategy': {
            'distribution': 'categorical',
            'values': ['entire', 'tcn_entire']
        },
        'lr': {
            'distribution': 'log_uniform',
            'min': -13.815510557964274,  # np.log(1e-6)
            'max': 0.0  # np.log(1)
        },
        'weight_decay': {
            'distribution': 'log_uniform',
            'min': -12.206072645530174,  # np.log(5e-6)
            'max': 0.0  # np.log(1)
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.9
        },
        'remove_disconnected_nodes': {
            'value': False
        },
        'channels_conv': {
            'value': 8
        },
        'normalisation': {
            'distribution': 'categorical',
            'values': ['subject_norm']
        },

        #### Values that change in each sweep
        # If Nones are not changed, wandb will have some errors
        'device': {
            'value': None
        },
        'fold_num': {
            'value': None
        },
        'num_nodes': {
            'value': None
        },
        'target_var': {
            'value': None
        },
        'num_epochs': {
            'value': None
        },
        'batch_size': {
            'value': None
        },
        'conn_type': {
            'value': None
        },
        'analysis_type': {
            'value': None
        },
        'time_length': {
            'value': None
        },
        'early_stop_steps': {
            'value': None
        },
        'sweep_type': {
            'value': None
        }
    }
}
