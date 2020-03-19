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
        'gcn_layers': {
            'distribution': 'categorical',
            'values': [0, 1, 2]
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
        }
    }
}