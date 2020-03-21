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
        'encoding': {
            'distribution': 'categorical',
            'values': ['none']
        },
        'conv_strategy': {
            'distribution': 'categorical',
            'values': ['entire', 'tcn_entire']
        },
        'pooling': {
            'distribution': 'categorical',
            'values': ['mean', 'diff_pool', 'concat']
        },
        'gnn_type': {
            'distribution': 'categorical',
            'values': ['gcn', 'none']
        },
        'gnn_layers': {
            'distribution': 'int_uniform',
            'min': 0,
            'max': 2
        },
        'threshold': {
            'distribution': 'categorical',
            'values': [5, 20]
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

        #### Values that change in each run
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
        }
    }
}
