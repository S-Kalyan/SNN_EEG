import sys

sys.path.append('../')
from mi_bci import RNNSpikeSignal
from framework_utils import *
import numpy as np

n_epochs = 10001
layer_wise_neurons = [600, 1500, 400]# number of nodes in each layer layers of spiking network
dt = 1e-3

class1_eeg = np.load('data/B0403TClass1_best.npy')
class2_eeg = np.load('data/B0403TClass2_best.npy')

desired_signal = [class1_eeg, class2_eeg]

n_steps = len(class1_eeg[0])
T = dt * n_steps
## assign number of noise neurons and noise factos
# noise_neurons = np.hstack((np.arange(10, 105, 5), np.arange(100, 260, 20)))
# noise_factor = np.hstack((1, np.arange(5, 45, 5)))
noise_neurons = np.hstack((np.arange(10, 105, 50), np.arange(100, 260, 50)))
noise_factor = np.hstack((1, np.arange(5, 45, 20)))
## generate and save artificial data for each combination
for nn in noise_neurons:
    for nf in noise_factor:
        model = RNNSpikeSignal(neurons=layer_wise_neurons, n_steps=n_steps, n_epochs=n_epochs,
                               desired_spike_signal=desired_signal, noise_neurons=nn, noise_factor=nf, freq=10,
                               lr=5e-4, syn_trials=100, last_epoch=n_epochs - 1, perturb=True)
        model.create_trials()
        print('completed_generating_noise_neuron: {} _noise_factor: {}'.format(nn, nf))
