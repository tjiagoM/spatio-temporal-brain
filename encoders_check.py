import numpy as np
import matplotlib.pyplot as plt

for person in [30, 60]:
    orig_ts = np.load(f'encoder_comparisons/0_{person}_orig.npy')
    recons_ts = np.load(f'encoder_comparisons/0_{person}_recons.npy')
    for i in range(50):
        fig, ax = plt.subplots(nrows=2, ncols=1)

        ax[0].plot(orig_ts[i, :])
        ax[0].set_title(f'Original - Person {person}, ICA {i}')
        ax[1].plot(recons_ts[i, :])
        ax[1].set_title(f'Reconstructed - Person {person}, ICA {i}')
        plt.tight_layout()
        plt.savefig(f'encoder_comparisons/fig_{person}_{i}.png')
        plt.close()
